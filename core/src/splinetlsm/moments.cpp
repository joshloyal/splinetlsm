#include <cmath>
#include <math.h>

#include "splinelsm.h"


namespace splinetlsm {
    
    inline arma::cube calculate_latent_position_means(
            const arma::cube& W, const sp_mat& B) {
        
        return W.each_slice() * B;
    }

    arma::cube calculate_latent_position_variances(
            const array4d& W_sigma, const sp_mat& B) {
        uint n_features = W_sigma.n_elem;
        uint n_nodes = W_sigma(0).n_rows;
        uint n_time_steps = B.n_cols;

        arma::cube U_sigma(n_nodes, n_time_steps, n_features);
        for (int d = 0; d < n_features_; ++d) {
            U_sigma.slice(d) = arma::sum(
                    (W_sigma(d).each_row() * B).each_row() % B, 2);
        }

        return U_sigma;
    }

    inline arma::mat calculate_coefs_means(
            const arma::mat& W_beta, const sp_mat& B) {
        return W_beta * B;
    }

    inline arma::mat calculate_coefs_variances(
            const arma::cube& W_beta_sigma, const sp_mat& B) {
        return arma::sum(
                (W_beta_sigma.each_row() * B).each_row() % B, 2);
    }

    arma::vec calculate_w_precisions(const ModelParams& params) {
        arma::vec w_prec(n_nodes);

        double theta = 0.;
        double eta_inv = 0.;
        for (int i = 0; i < n_nodes; ++i) {
            theta = std::sqrt(params.a * params.b(i));
            eta_inv = std::sqrt(params.a / params.b(i));
            
            w_prec(i) = (eta_inv * (
                std::cyl_bessel_k(params.p + 1, theta) / 
                    std::cyl_bessel_k(params.p, theta));
            w_prec(i) -= (2 * params.p) / params.b(i);
        }

        return w_prec;
    }

    arma::vec calculate_w_coefs_precisions(const ModelParams& params) {
        uint n_covariates = params.W_coefs.n_rows;
        arma::vec w_coefs_prec(n_covariates);

        double theta = 0.;
        double eta_inv = 0.;
        for (int k = 0; k < n_covariates; ++k) {
            theta = std::sqrt(params.a_coefs * params.b_coefs(k));
            eta_inv = std::sqrt(params.a_coefs / params.b_coefs(k));
            
            w_coefs_prec(k) = (eta_inv * (
                std::cyl_bessel_k(params.p_coefs + 1, theta) / 
                    std::cyl_bessel_k(params.p_coefs, theta)));
            w_coefs_prec(k) -= (2 * params.p_coefs) / params.b_coefs(k);
        }

        return w_coefs_prec;
    }

    inline arma::vec calculate_log_gamma_means(arma::vec& shape, arma::vec& rate) {
        return arma::cumsum(log(shape(h)) - log(rate(h)));
    }

    Moments calculate_moments(const ModelParams& params, const sp_mat& B) {
        Moments moments;
        
        // moments of the latent positions
        moments.U = calculate_latent_position_means(params.W, B);
        moments.U_sigma = calculate_latent_position_variances(
                params.W_sigma, B);

        // moments of the time-varying coefficients
        moments.coefs = calculate_coefs_mean(params.W_beta, B);
        moments.coefs_sigma = calculate_coefs_sigma(params.W_beta_sigma, B);
        
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
