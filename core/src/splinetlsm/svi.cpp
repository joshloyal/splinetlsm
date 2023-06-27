#include <cmath>
#include <utility>

#include "splinetlsm.h"


namespace splinetlsm {
    
    SVI::SVI(ModelConfig& config, double step_size_delay=1, 
            double step_size_power=0.75, double tol=0.001) :  
        converged(false),
        config_(config),
        iter_idx_(0), 
        step_size_delay_(step_size_delay), 
        step_size_power_(step_size_power), 
        tol_(tol) {}


    std::pair<NaturalParams, ModelParams>
    SVI::update(const sp_cube& Y, const sp_mat& B, const array4d& X,
            arma::uvec& time_points, NaturalParams& natural_params, 
            ModelParams& params) {
        // Y : sparse csc cube of shape T x n x n
        // B : sparse csc matrix of shape L_M x T
        uint n_nodes = Y.n_cols;

        // initialize new natural parameters
        new_natural_params = NaturalParams(config_);
        
        // update step size
        iter_idx_ += 1
        double step_size = 1. / pow(
            iter_idx_ + step_size_delay_, step_size_power_); 

        //------- Sample Dyads ----------------------------------------------//
        
        // sub-sample time indices and dyads
        SampleInfo sample_info = dyad_sampler.draw(Y, time_points);
        
        // the b-spline design matrix evaluated at the sampled snapshots
        // XXX: To access observation quantities (Y, X) for time sample number t
        //      use Y(time_indices(t))
        arma::sp_mat B_sub = B.cols(sample_info.time_indices);

        // evaluate means and covariances at the sampled time points only
        Moments moments = calculate_moments(params, B_sub);


        //------- Optimize Local Variables ----------------------------------//
        
        // update q(omega_ijt) for sampled dyads
        arma::field<arma::vec> omega(sample_info.time_indices.n_elem, n_nodes);
        for (int t = 0; t < sample_info.time_indices.n_elem; ++t) {
            for (int i = 0; i < n_nodes; ++i) {
                arma::uvec dyads = sample_info.dyad_indices(t, i);
                omega(t, i) = arma::vec(dyads.n_elem);
                uint dyad_idx = 0;
                for (auto j : dyads) {
                    omega(t, i)(dyad_idx) = calculate_omega(moments, X, i, j, t);
                    dyad_idx += 1;
                }
            }
        }
       

        //------- Latent Position Spline Weights ----------------------------//
        
        // update latent position weights: q(w_ih)
        arma::mat prior_precision;
        for (int i = 0; i < n_nodes; i++) {
            
            // calculate prior precision matrix for node i (Omega_i)
            prior_precision = moments.w_prec(i) * config_.penalty_matrix;
            for (int r = 0; r < config_.penality_order; ++r) {
                prior_precision(r, r) += config_.tau_prec;
            }

            for (int h = 0; h < n_features; h++) {
                auto [grad_mean, grad_prec] = calculate_latent_position_gradients(
                        Y, X, B_sub, moments, prior_precision, omega,
                        sample_info, i, h);
                
                // take a gradient step
                new_natural_params.W(i, h) = (
                        (1 - step_size) * natural_params.W(i, h). + 
                            step_size * grad_mean);

                new_natural_params.W_sigma(h).row(i) = (
                        (1 - step_size) * natural_params.W_sigma(h).row(i) + 
                            step_size * grad_prec);
            }
        }


        //------- Time-Varying Coefficient Spline Weig-----------------------//
        
        // update covariate weights: q(w_k)
        arma::mat prior_precision;
        for (int k = 0; k < n_covariates; k++) {
            // calculate prior precision matrix for feature k
            prior_precision = (
                moments.w_coefs_prec(k) * config_.coefs_penalty_matrix);

            for (int r = 0; r < config_.penality_order; ++r) {
                prior_precision(r, r) += config_.tau_coefs_prec;
            }
            
            auto [grad_mean, grad_prec] = calculate_coef_gradients(
                    Y, X, B_sub, moments, prior_precision, omega,
                    sample_info, k);
                
            // take a gradient step
            new_natural_params.W_coefs(k) = (
                    (1 - step_size) * natural_params.W_coefs(k) + 
                        step_size * grad_mean);

            new_natural_params.W_coefs.row(k) = (
                    (1 - step_size) * natural_params.W_coefs_sigma.row(k) +
                        step_size * grad_prec);
        }
        
        //------- High-Level Variance Parameters --------------------------//
        
        // update node variances: q(sigma_i)
        for (int i = 0; i < n_nodes; ++i) {
            double grad_b = calculate_node_variance_gradient(
                params.W, params.W_sigma, params.log_gamma,
                config_.diff_matrix, config_.penalty_matrix);

            new_natural_params.b(i) = (
                (1 - step_size) * natural_params.b(i) + step_size * grad_b);
        }

        // update covariate variances: q(sigma_k)
        for (int k = 0; k < n_covariates; ++k) {
            double grad_b = calculate_coef_variance_gradient(
                params.W_coefs, params.W_coefs_sigma,
                config_.coefs_diff_matrix, config_.coefs_penalty_matrix, k);

            
            new_natural_params.b_coefs(k) = (
                (1 - step_size) * natural_params.b_coefs(k) + 
                    step_size * grad_b);
        }

        // update MGP parameters: q(nu_h)
        for (int h = 0; h < config_.n_features; ++h) {

            double grad_rate = calculate_mgp_variance_gradient(
                params.W, params.W_sigma, params.mgp_rate, params.mgp_shape, 
                moments.log_gamma, moments.w_prec, 
                config_.tau_prec, config_.diff_matrix,
                config_.penalty_matrix_, config_.penalty_order_);

            new_natural_params.mgp_rate(h) = (
                (1 - step_size) * natural_params.mgp_rate(h) + 
                    step_size * grad_d);
        }

        
        // transform natural parameters to standard parameters
        ModelParams new_params(new_natural_params);


        //------- Check for Convergence -------------------------------------//

        // XXX: - overloaded to calculate the absolute value 
        //      of the difference of natural parameters
        natural_params_diff = new_natural_params - natural_params;
        if (natural_params_diff < tol_) {
            converged = true;
        }

        return {new_natural_params, new_params};
    }


    ModelParams optimize_elbo(
            const sp_cube& Y, const sp_mat& B, const array4d& X,
            arma::uvec& time_points, uint n_features=2,
            uint penalty_order=1, uint coefs_penalty_order=1, 
            double rate_prior=2, double shape_prior=1,
            double coefs_rate_prior=2, double coefs_shape_prior=1,
            double mgp_a1=2, double mgp_a2=3,
            double tau_prec=0.01, double tau_coefs_prec=0.01,
            double step_size_delay=1, double step_size_power=0.75,
            uint max_iter=100, double tol=0.001, int random_state=42) {
        
        // set random seed
        arma_rng::set_seed(random_state); 

        config = ModelConfig(Y, B, X, n_features, penalty_order, 
            coefs_penalty_order, rate_prior, shape_prior,
            coefs_rate_prior, coefs_shape_prior, mgp_a1, mgp_a2,
            tau_prec, coefs_tau_prec);

        svi = SVI(config, step_size_delay, step_size_power, tol);
        
        // initial parameter values
        natural_params = NaturalParams(config);
        params = ModelParams(natural_params);
        
        for (uint iter = 0; iter < max_iter; iter++) {
            auto [natural_params, params] = svi.update(
                    Y, B, X, time_points, natural_params, params);
        
            // check for convergence
            if (svi.converged) {
                break;
            }
        }
        
        return params;
    }
}
