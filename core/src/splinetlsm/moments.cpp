#include <cmath>
#include <math.h>
#include <boost/math/special_functions/bessel.hpp>

#include "splinetlsm.h"


namespace splinetlsm {
    
    arma::cube calculate_latent_position_means(
            const arma::cube& W, const arma::sp_mat& B) {
        
        // XXX: not supported for sparse matrices...
        // U = W.each_slice() * B;
        
        arma::cube U(W.n_rows, B.n_cols, W.n_slices);
        for (uint h = 0; h < W.n_slices; ++h) {
            U.slice(h) = W.slice(h) * B;
        }

        return U;
    }

    arma::cube calculate_latent_position_variances(
            const array4d& W_sigma, const arma::sp_mat& B) {
        uint n_features = W_sigma.n_elem;
        uint n_nodes = W_sigma(0).n_slices;
        uint n_time_steps = B.n_cols;

        arma::cube U_sigma(n_nodes, n_time_steps, n_features);
        for (uint h = 0; h < n_features; ++h) {

            // XXX: each_slice does not support sparse matrix operations
            // U_sigma.slice(h) = arma::sum(
            //        (W_sigma(h).each_slice() * B).each_slice() % B, 0).t();
            
            for (uint i = 0; i < n_nodes; ++i) {
                U_sigma.slice(h).row(i) = arma::sum(
                        (W_sigma(h).slice(i) * B) % B, 0);
            }
        }

        return U_sigma;
    }

    inline arma::mat calculate_coefs_means(
            const arma::mat& W_coefs, const arma::sp_mat& B) {
        return W_coefs.t() * B;
    }

    arma::mat calculate_coefs_variances(
            const arma::cube& W_coefs_sigma, const arma::sp_mat& B) {
 
        uint n_covariates = W_coefs_sigma.n_slices;
        uint n_time_steps = B.n_cols;
        
        // XXX: each_slice does not support sparse matrix apporations
        // coefs_sigma = arma::sum(
        // (W_coefs_sigma.each_slice() * B).each_slice() % B, 0);
        
        arma::mat coefs_sigma(n_covariates, n_time_steps);
        for (uint k = 0; k < n_covariates; ++k) {
            coefs_sigma.row(k) = arma::sum((W_coefs_sigma.slice(k) * B) % B, 0);
        }
        
        return coefs_sigma;
    }

    arma::vec calculate_w_precisions(const ModelParams& params) {
        uint n_nodes = params.b.n_elem;

        arma::vec w_prec(n_nodes);

        double theta = 0.;
        double eta_inv = 0.;
        for (uint i = 0; i < n_nodes; ++i) {
            theta = std::sqrt(params.a * params.b(i));
            eta_inv = std::sqrt(params.a / params.b(i));
            
            w_prec(i) = (eta_inv * (
                boost::math::cyl_bessel_k(params.p + 1, theta) / 
                    boost::math::cyl_bessel_k(params.p, theta)));
            w_prec(i) -= (2 * params.p) / params.b(i);
        }

        return w_prec;
    }

    arma::vec calculate_w_coefs_precisions(const ModelParams& params) {
        uint n_covariates = params.W_coefs.n_cols;
        arma::vec w_coefs_prec(n_covariates);

        double theta = 0.;
        double eta_inv = 0.;
        for (uint k = 0; k < n_covariates; ++k) {
            theta = std::sqrt(params.a_coefs * params.b_coefs(k));
            eta_inv = std::sqrt(params.a_coefs / params.b_coefs(k));
            
            w_coefs_prec(k) = (eta_inv * (
                boost::math::cyl_bessel_k(params.p_coefs + 1, theta) / 
                    boost::math::cyl_bessel_k(params.p_coefs, theta)));
            w_coefs_prec(k) -= (2 * params.p_coefs) / params.b_coefs(k);
        }

        return w_coefs_prec;
    }

    inline arma::vec calculate_log_gamma_means(
            const arma::vec& shape, const arma::vec& rate) {
        return arma::cumsum(log(shape) - log(rate));
    }

    Moments calculate_moments(
            const ModelParams& params, const arma::sp_mat& B) {
        Moments moments;
        
        // moments of the latent positions
        moments.U = calculate_latent_position_means(params.W, B);
        moments.U_sigma = calculate_latent_position_variances(
                params.W_sigma, B);

        // moments of the time-varying coefficients
        moments.coefs = calculate_coefs_means(params.W_coefs, B);
        moments.coefs_sigma = calculate_coefs_variances(
                params.W_coefs_sigma, B);
        
        // latent position precisions E_q[1/sigma_i^2]
        moments.w_prec = calculate_w_precisions(params);
        
        // coefs precisions E_q[1/sigma_beta_k^2]
        moments.w_coefs_prec = calculate_w_coefs_precisions(params);
        
        // log means of MGP variances: log(E_q[gamma_h])
        moments.log_gamma = calculate_log_gamma_means(
            params.mgp_shape, params.mgp_rate);

        return moments;
    }
}
