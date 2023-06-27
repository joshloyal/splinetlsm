#include <cmath>

#include "splinetlsm.h"


namespace splinetlsm {

    double calculate_omega(Moments& moments, array4d& X, uint i, uint j, uint t,
            SampleInfo& sample_info) {
        
        uint time_index = sample_info.time_indices(t);

        // extract necessary parameters
        arma::vec x = X(time_index).tube(i, j);
        arma::vec mu_i = moments.U.tube(i, t);
        arma::vec mu_j = moments.U.tube(j, t);
        arma::vec mu_beta = moments.coefs.col(t);
        arma::vec Sigma_beta = moments.coefs_sigma.col(t);
        arma::vec Sigma_i = moments.U_sigma.tube(i, t);
        arma::vec Sigma_j = moments.U_sigma.tube(i, t);
        
        // calculate mu_{omega_{ij,t}}
        double c_sq = pow(mu_beta.t() * x + mu_i.t() * mu_j, 2);
        
        //c_sq += arma::accu((x * x.t()) % Sigma_beta);
        c_sq += arma::accu(x.t() * (Sigma_beta % x));
        
        c_sq += Sigma_i.t() * Sigma_j;
        c_sq += mu_i.t() * (Sigma_j % mu_i);
        c_sq += mu_j.t() * (Sigma_j % mu_j);
        
        // calculate mean of PG(1, c)
        double c = sqrt(c_sq); 
        return 0.5 * (1. / c) * tanh(0.5 * c);
    }

}
