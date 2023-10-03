#include "splinetlsm.h"


namespace splinetlsm {

    ModelConfig::ModelConfig(
            const sp_cube& Y, const arma::sp_mat& B, const array4d& X, 
            uint n_features, uint penalty_order, uint coefs_penalty_order, 
            double rate_prior, double shape_prior,
            double coefs_rate_prior, double coefs_shape_prior,
            double mgp_a1, double mgp_a2,
            double tau_prec, double coefs_tau_prec, double alpha) :
        n_time_points(Y.n_elem),
        n_nodes(Y(0).n_rows),
        n_features(n_features),
        n_covariates(X(0).n_slices + 1),  // intercept + covariates
        n_knots(B.n_rows),
        penalty_order(penalty_order),
        coefs_penalty_order(coefs_penalty_order),
        rate_prior(rate_prior),
        shape_prior(shape_prior),
        coefs_rate_prior(coefs_rate_prior),
        coefs_shape_prior(coefs_shape_prior),
        mgp_a1(mgp_a1), 
        mgp_a2(mgp_a2),
        tau_prec(tau_prec), 
        coefs_tau_prec(coefs_tau_prec),
        alpha(alpha),
        density(n_time_points) {

            // latent positions' difference and penalty matrices
            diff_matrix = diff(
                arma::mat(n_knots, n_knots, arma::fill::eye), penalty_order);
            penalty_matrix = diff_matrix.t() * diff_matrix;

            // coefficients' difference and penalty matrices
            coefs_diff_matrix = diff(
                arma::mat(n_knots, n_knots, arma::fill::eye), coefs_penalty_order);
            coefs_penalty_matrix = coefs_diff_matrix.t() * coefs_diff_matrix;

            for (uint h = 0; h < n_time_points; ++h) {
                density(h) = arma::mean(arma::mean(Y(h)));
            }
    }
}
