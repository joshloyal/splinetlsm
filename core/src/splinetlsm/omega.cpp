#include <cmath>

#include "splinetlsm.h"


namespace splinetlsm {

    double calculate_omega(Moments& moments, const arma::cube& X, uint i, uint j, 
            uint t) {
        
        //uint time_index = sample_info.time_indices(t);

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

}
