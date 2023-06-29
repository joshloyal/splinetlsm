#include <cmath>
#include <utility>

#include "splinetlsm.h"


namespace splinetlsm {
     
    SVI::SVI(ModelConfig& config, double nonedge_proportion,
            uint n_time_steps,
            double step_size_delay, 
            double step_size_power, double tol) :  
        converged(false),
        config_(config),
        dyad_sampler_(nonedge_proportion, n_time_steps),
        iter_idx_(0), 
        step_size_delay_(step_size_delay), 
        step_size_power_(step_size_power), 
        tol_(tol) {}


    Params
    SVI::update(const sp_cube& Y, const arma::sp_mat& B, const array4d& X,
                const arma::vec& time_points, Params& params) {
        // Y : sparse csc cube of shape T x n x n
        // B : sparse csc matrix of shape L_M x T
        uint n_nodes = Y(0).n_cols;

        // initialize new natural parameters
        NaturalParams new_natural_params(config_);

        // update step size
        iter_idx_ += 1;
        double step_size = 1. / pow(
            iter_idx_ + step_size_delay_, step_size_power_); 

        //------- Sample Dyads ----------------------------------------------//
        
        // sub-sample time indices and dyads
        SampleInfo sample_info = dyad_sampler_.draw(Y, time_points);
        
        // the b-spline design matrix evaluated at the sampled snapshots
        // XXX: To access observation quantities (Y, X) for time sample number t
        //      use Y(time_indices(t))
        arma::sp_mat B_sub = B.cols(sample_info.time_indices);
        
        // evaluate means and covariances at the sampled time points only
        Moments moments = calculate_moments(params.model, B_sub);

        //------- Optimize Local Variables ----------------------------------//
        
        // update q(omega_ijt) for sampled dyads
        arma::field<arma::vec> omega = optimize_omega(moments, X, sample_info);
 

        //------- Latent Position Spline Weights ----------------------------//
        
        // update latent position weights: q(w_ih)
        for (uint i = 0; i < n_nodes; i++) {
            
            // calculate prior precision matrix for node i (Omega_i)
            arma::mat prior_precision = calculate_prior_precision(
                    moments.w_prec(i), config_);

            for (uint h = 0; h < config_.n_features; h++) {
                auto [grad_mean, grad_prec] = calculate_latent_position_gradients(
                        Y, X, B_sub, moments, prior_precision, omega,
                        sample_info, i, h);
                
                // take a gradient step
                new_natural_params.W.slice(h).row(i) = (
                        (1 - step_size) * params.natural.W.slice(h).row(i) + 
                            step_size * grad_mean.t());

                new_natural_params.W_sigma(h).slice(i) = (
                        (1 - step_size) * params.natural.W_sigma(h).slice(i) + 
                            step_size * grad_prec);
            }
        }


        //------- Time-Varying Coefficient Spline Weights -------------------//
        
        // update covariate weights: q(w_k)
        for (uint k = 0; k < config_.n_covariates; k++) {
            // calculate prior precision matrix for feature k
            arma::mat prior_precision = calculate_coefs_prior_precision(
                    moments.w_coefs_prec(k), config_);
            
            auto [grad_mean, grad_prec] = calculate_coef_gradients(
                    Y, X, B_sub, moments, prior_precision, omega,
                    sample_info, k);
                
            // take a gradient step
            new_natural_params.W_coefs.col(k) = (
                    (1 - step_size) * params.natural.W_coefs.col(k) + 
                        step_size * grad_mean);

            new_natural_params.W_coefs_sigma.slice(k) = (
                    (1 - step_size) * params.natural.W_coefs_sigma.slice(k) +
                        step_size * grad_prec);
        }
        
        //------- High-Level Variance Parameters --------------------------//
        
        // update node variances: q(sigma_i)
        for (uint i = 0; i < n_nodes; ++i) {
            double grad_b = calculate_node_variance_gradient(
                params.model.W, params.model.W_sigma, moments.log_gamma,
                config_.diff_matrix, config_.penalty_matrix, i);

            new_natural_params.b(i) = (
                (1 - step_size) * params.natural.b(i) + step_size * grad_b);
        }

        // update covariate variances: q(sigma_k)
        for (uint k = 0; k < config_.n_covariates; ++k) {
            double grad_b = calculate_coef_variance_gradient(
                params.model.W_coefs, params.model.W_coefs_sigma,
                config_.coefs_diff_matrix, config_.coefs_penalty_matrix, k);

            
            new_natural_params.b_coefs(k) = (
                (1 - step_size) * params.natural.b_coefs(k) + 
                    step_size * grad_b);
        }

        // update MGP parameters: q(nu_h)
        for (uint h = 0; h < config_.n_features; ++h) {

            double grad_rate = calculate_mgp_variance_gradient(
                params.model.W, params.model.W_sigma, params.model.mgp_rate, 
                params.model.mgp_shape, moments.log_gamma, moments.w_prec, 
                config_.tau_prec, config_.diff_matrix,
                config_.penalty_matrix, config_.penalty_order, h);

            new_natural_params.mgp_rate(h) = (
                (1 - step_size) * params.natural.mgp_rate(h) + 
                    step_size * grad_rate);
        }

        
        // transform natural parameters to standard parameters
        ModelParams new_params(new_natural_params);

        //------- Check for Convergence -------------------------------------//

        // XXX: - overloaded to calculate the absolute value 
        //      of the difference of natural parameters
        double natural_params_diff = new_natural_params - params.natural;
        if (natural_params_diff < tol_) {
            converged = true;
        }

        return {new_natural_params, new_params};
    }


    ModelParams optimize_elbo(
            const sp_cube& Y, const arma::sp_mat& B, const array4d& X,
            const arma::vec& time_points, uint n_features,
            uint penalty_order, uint coefs_penalty_order, 
            double rate_prior, double shape_prior,
            double coefs_rate_prior, double coefs_shape_prior,
            double mgp_a1, double mgp_a2,
            double tau_prec, double coefs_tau_prec,
            double nonedge_proportion, uint n_time_steps,
            double step_size_delay, double step_size_power,
            uint max_iter, double tol, int random_state) {
        
        // set random seed
        arma::arma_rng::set_seed(random_state); 
        
        // store various hyperparameters and statistics
        ModelConfig config(Y, B, X, n_features, penalty_order, 
            coefs_penalty_order, rate_prior, shape_prior,
            coefs_rate_prior, coefs_shape_prior, 
            mgp_a1, mgp_a2, tau_prec, coefs_tau_prec);
        
        // initialize SVI algorithm
        SVI svi(config, nonedge_proportion, n_time_steps, 
                step_size_delay, step_size_power, tol);
        
        // initial parameter values
        Params params(config);
        
        // run stochastic gradient descent
        for (uint iter = 0; iter < max_iter; ++iter) {
            Params new_params = svi.update(Y, B, X, time_points, params);
            
            Params params = new_params;

            // check for convergence
            if (svi.converged) {
                break;
            }
        }
        
        return params.model;
    }
}
