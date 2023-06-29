#pragma once

namespace splinetlsm {
    
    // moments derived from the variational distribution parameters
    struct Moments {
        arma::cube U;               // n x T_sub x d array of latent position means E[u_{ih}(t_m)]
        arma::cube U_sigma;         // n x T_sub x d array of latent position variances Var(u_{ih}(t_m)). 
                                    // (NOTE: q has no covariance between dimensions h and h')
        arma::mat coefs;            // p x T_sub matrix of time-varying coefficients E[beta_p(t)]
        arma::mat coefs_sigma;      // p x T_sub matrix of time-varying coefficients variances Var(beta_p(t))
                                    // (NOTE: q has no covariance between covariates)
        arma::vec w_prec;           // n-dimensional expected precisions E[1/sigma_i^2]
        arma::vec w_coefs_prec;     // p-dimensional expected precisions E[1/sigma_beta_k^2]

        arma::vec log_gamma;        // d-dimensional MGP log-precisions log(E[gamma_h])
    };
    
    arma::mat calculate_prior_precision(double w_prec, ModelConfig& config);
    arma::mat calculate_coefs_prior_precision(
            double w_prec, ModelConfig& config);
    
    arma::cube calculate_latent_position_means(
            const arma::cube& W, const arma::sp_mat& B);
    
    arma::cube calculate_latent_position_variances(
            const array4d& W_sigma, const arma::sp_mat& B);
    
    inline arma::mat calculate_coefs_means(
            const arma::mat& W_coefs, const arma::sp_mat& B);
    
    arma::mat calculate_coefs_variances(
            const arma::cube& W_coefs_sigma, const arma::sp_mat& B);
    
    arma::vec calculate_w_precisions(const ModelParams& params);
    
    arma::vec calculate_w_coefs_precisions(const ModelParams& params);
    
    inline arma::vec calculate_log_gamma_means(
            const arma::vec& shape, const arma::vec& rate);
    
    Moments calculate_moments(const ModelParams& params, const arma::sp_mat& B);

}
