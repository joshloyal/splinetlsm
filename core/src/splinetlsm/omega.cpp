#include <cmath>

#include "splinetlsm.h"


namespace splinetlsm {

    double optimize_omega_single(const Moments& moments, const arma::cube& X, 
            uint i, uint j, uint t) {
        
        // extract necessary parameters
        arma::vec x = X.tube(i, j);
        arma::vec mu_i = moments.U.tube(i, t);
        arma::vec mu_j = moments.U.tube(j, t);
        arma::vec mu_coefs = moments.coefs.col(t);
        arma::vec Sigma_coefs = moments.coefs_sigma.col(t);
        arma::vec Sigma_i = moments.U_sigma.tube(i, t);
        arma::vec Sigma_j = moments.U_sigma.tube(i, t);
        
        // calculate mu_{omega_{ij,t}}
        double c_sq = pow(
                arma::as_scalar(mu_coefs.t() * x + mu_i.t() * mu_j), 2);
        
        //c_sq += arma::accu((x * x.t()) % Sigma_beta);
        c_sq += arma::accu(x.t() * (Sigma_coefs % x));
        
        c_sq += arma::as_scalar(Sigma_i.t() * Sigma_j);
        c_sq += arma::as_scalar(mu_i.t() * (Sigma_j % mu_i));
        c_sq += arma::as_scalar(mu_j.t() * (Sigma_j % mu_j));
        
        // calculate mean of PG(1, c)
        double c = sqrt(c_sq); 
        return 0.5 * (1. / c) * tanh(0.5 * c);
    }
    
    arma::field<arma::vec> optimize_omega(
            const Moments& moments, const array4d& X, 
            const SampleInfo& sample_info) {
        
        uint n_nodes = X(0).n_rows;

        arma::field<arma::vec> omega(sample_info.time_indices.n_elem, n_nodes);
        for (uint t = 0; t < sample_info.time_indices.n_elem; ++t) {
            // extract covariates at time t
            arma::cube Xt = X(sample_info.time_indices(t));

            for (uint i = 0; i < n_nodes; ++i) {
                arma::uvec dyads = sample_info.dyad_indices(t, i);
                omega(t, i) = arma::vec(dyads.n_elem);
                uint dyad_idx = 0;
                for (auto j : dyads) {
                    omega(t, i)(dyad_idx) = optimize_omega_single(
                            moments, Xt, i, j, t);
                    dyad_idx += 1;
                }
            }
        }

        return omega;
    }
}
