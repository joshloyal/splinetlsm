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
        arma::vec w_beta_prec;      // p-dimensional expected precisions E[1/sigma_beta_k^2]

        arma::vec log_gamma;        // d-dimensional MGP log-precisions log(E[gamma_h])
    };
    
    inline arma::cube calculate_latent_position_means(
            const arma::cube& W, const sp_mat& B);
    
    arma::cube calculate_latent_position_variances(
            const array4d& W_sigma, const sp_mat& B);
    
    inline arma::mat calculate_coefs_means(
            const arma::mat& W_beta, const sp_mat& B);
    
    inline arma::mat calculate_coefs_variances(
            const arma::cube& W_beta_sigma, const sp_mat& B);
    
    arma::vec calculate_w_precisions(const ModelParams& params);
    
    arma::vec calculate_w_coefs_precisions(const ModelParams& params);
    
    inline arma::vec calculate_log_gamma_means(
            arma::vec& shape, arma::vec& rate);
    
    Moments calculate_moments(const ModelParams& params, const sp_mat& B);

}
