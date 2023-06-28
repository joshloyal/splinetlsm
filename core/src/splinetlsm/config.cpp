#include "splinetlsm.h"


namespace splinetlsm {

    ModelConfig::ModelConfig(
            const sp_cube& Y, const arma::sp_mat& B, const array4d& X, 
            uint n_features, uint penalty_order, uint coefs_penalty_order, 
            double rate_prior, double shape_prior,
            double coefs_rate_prior, double coefs_shape_prior,
            double mgp_a1, double mgp_a2,
            double tau_prec, double coefs_tau_prec) :
            //uint n_features=2, uint penalty_order=1, uint coefs_penalty_order=1, 
            //double rate_prior=2, double shape_prior=1,
            //double coefs_rate_prior=2, double coefs_shape_prior=1,
            //double mgp_a1=2, double mgp_a2=3,
            //double tau_prec=0.01, double tau_coefs_prec=0.01) :
        n_nodes(Y(0).n_rows),
        n_features(n_features),
        n_covariates(X(0).n_slices),
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
        coefs_tau_prec(coefs_tau_prec) {

            // latent positions' difference and penalty matrices
            diff_matrix = diff(
                arma::mat(n_knots, n_knots, arma::fill::eye), penalty_order);
            penalty_matrix = diff_matrix.t() * diff_matrix;

            // coefficients' difference and penalty matrices
            coefs_diff_matrix = diff(
                arma::mat(n_knots, n_knots, arma::fill::eye), coefs_penalty_order);
            coefs_penalty_matrix = coefs_diff_matrix.t() * coefs_diff_matrix;

    } 
}
