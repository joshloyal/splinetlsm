#include <cmath>
#include <utility>

#include "splinetlsm.h"


namespace splinetlsm {

    std::pair<arma::vec, arma::mat> calculate_latent_position_gradients(
            const sp_cube& Y, const array4d& X, const sp_mat& B, 
            Moments& moments, arma::mat& prior_precision, 
            arma::field<arma::vec>& omega, 
            SampleInfo& sample_info, uint i, uint h) {
        
        // initialize gradients
        arma::vec grad_mean(B.n_rows, arma::fill:zeros);
        arma::mat grad_prec(B.n_rows, B.n_rows, arma::fill:zeros);

        for (uint t = 0; t < sample_info.time_indices.n_elem; ++t) {
            // reset indices and weights
            double time_weight_mean = 0; 
            double time_weight_prec = 0; 
            uint dyad_idx = 0.;
            
            // necessary values to calculate gradients
            uint time_index = sample_info.time_indices(t);
            arma::vec omega_it = omega(t, i);
            arma::vec mu_i = moments.U.tube(i, t);
            arma::vec coefs = moments.coefs.col(t);
            double degree = sample_info.degree(t, i);
            double nonedge_weight = sample_info.weights(t, i);

            for (auto j : sample_info.dyad_indices(t, i)) {
                // get necessary variables to calculate gradients
                double z = Y(time_index, i, j) - 0.5;
                arma::vec x = X(time_index).tube(j, t);
                arma::vec mu_j = moments.U.tube(j, t);
                double sigma_jh = moments.U_sigma(j, t, h);
                
                // dyad weight
                double sample_weight = (dyad_idx < degree) ? 1.0 : nonedge_weight;
                
                // gradient of the mean
                double residual = z - omega_it(dyad_idx) * (
                        coefs.t() * x + mu_i.t() * mu_j - mu_i(h) * mu_j(h));

                time_weight_mean += sample_weight * mu_j(h) * residual; 

                // gradient of the precision
                time_weight_prec += sample_weight * omega_it(dyad_idx) * (
                        pow(mu_j(h), 2) + sigma_jh)

                dyad_idx += 1;
            }

            grad_mean += time_weight_mean * B.col(t);
            grad_prec += time_weight_prec * (B.col(t) * B.col(t).t())
        }
        
        // re-weight sample for sampling of time points
        double time_weight = Y.n_elem / sample_info.time_indices.n_elem;
        grad_mean *= time_weight;
        grad_prec *= time_weight;

        // add the prior precision to grad_prec
        grad_prec += moments.gamma(h) * prior_precision;
        

        return {grad_mean, grad_prec};
    }
 
}
