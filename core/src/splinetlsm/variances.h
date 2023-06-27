#pragma once

namespace splinetlsm {
    
    double calculate_expected_penalty(arma::vec& w, arma::mat& w_sigma,
            arma::mat& diff_matrix, arma::mat& penalty_matrix);
    
    double calculate_node_variance_gradient(
             arma::cube& W, array4d& W_sigma, arma::vec& log_gamma,
             arma::mat& diff_matrix, arma::mat& penalty_matrix);
    
    inline double calculate_coef_variance_gradient(
            arma::mat& W_coefs, arma::cube& W_coefs_sigma,
            arma::mat& diff_matrix, arma::mat& penalty_matrix, uint k);
    
    double calculate_mgp_variance_gradient(
            arma::cube& W, array4d& W_sigma, 
            arma::vec& mgp_rate, arma::vec& mgp_shape, arma::vec& log_gamma, 
            arma::vec& w_prec, 
            double tau_prec, arma::mat& diff_matrix, 
            arma::mat& penalty_matrix, uint penalty_order);
}
