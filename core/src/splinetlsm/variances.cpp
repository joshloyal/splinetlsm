#include <cmath>

#include "splinetlsm.h"


namespace splinetlsm {
    
    double calculate_expected_penalty(arma::vec& w, arma::mat& w_sigma,
            arma::mat& diff_matrix, arma::mat& penalty_matrix) {
        arma::vec diff = diff_matrix * w;
        return diff.t() * diff + arma::accu(penalty_matrix % w_sigma);
    }

    double calculate_node_variance_gradient(
             arma::cube& W, array4d& W_sigma, arma::vec& log_gamma,
             arma::mat& diff_matrix, arma::mat& penalty_matrix) {
        
        uint n_features = W.n_slices;
        double grad_b = 0.;
        for (int h = 0; h < n_features; ++h) {
            grad_b += exp(log_gamma(h)) * calculate_expected_penalty(
                W.row(i).col(h), W_sigma(h).row(i), diff_matrix, 
                penalty_matrix);
        }

        return grad_b;
    }

    inline double calculate_coef_variance_gradient(
            arma::mat& W_coefs, arma::cube& W_coefs_sigma,
            arma::mat& diff_matrix, arma::mat& penalty_matrix, uint k) {
     
        return calculate_expected_penalty(
            W_coefs.rows(k).t(), W_coefs_sigma.row(k), diff_matrix, 
            penalty_matrix);
    }

    double calculate_mgp_variance_gradient(
            arma::cube& W, array4d& W_sigma, 
            arma::vec& mgp_rate, arma::vec& mgp_shape, arma::vec& log_gamma, 
            arma::vec& w_prec, 
            double tau_prec, arma::mat& diff_matrix, 
            arma::mat& penalty_matrix, uint penalty_order) {
        
        uint n_nodes = W.n_rows;
        uint n_features = W.n_slices;

        double grad_d = 2.;
        for (uint l = h; l < n_features; ++l) {
            double grad_lh = 0.
            for (uint i = 0; i < n_nodes; ++i) {
                grad_lh += w_prec(i) * calculate_expected_penalty(
                        W.row(i).col(l), W_sigma(h).row(l), diff_matrix,
                        penalty_matrix);

                for (uint r = 0; r < penalty_order; ++r) {
                   grad_lh += tau_prec * (
                           pow(W(i).col(h)(r), 2) + W_sigma(h).row(i)(r, r));
                }
            }

            double gamma_lh = exp(log_gamma(l) - (mgp_shape(h) - mgp_rate(h)))
            grad_d += gamma_lh * grad_lh;
        }

        return grad_d;
    }

}
