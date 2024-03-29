#include <cmath>

#include "splinetlsm.h"


namespace splinetlsm {
    
    arma::vec get_covariates(const arma::cube& X, uint i, uint j) {
        arma::vec x = arma::vec(1, arma::fill::ones);
        if (X.n_slices > 0) {
            arma::vec x_covariates = X.tube(i, j);
            x = arma::join_cols(x, x_covariates);
        }

        return x;
    }

    std::pair<arma::vec, arma::mat> calculate_coef_gradients(
            const sp_cube& Y, const array4d& X, const arma::sp_mat& B, 
            Moments& moments, arma::mat& prior_precision, 
            arma::field<arma::vec>& omega, double alpha,
            SampleInfo& sample_info, uint k) {
        
        uint n_nodes = Y(0).n_rows;
        uint n_time_steps = sample_info.time_indices.n_elem;

        // initialize gradients
        arma::vec grad_mean(B.n_rows, arma::fill::zeros);
        arma::mat grad_prec(B.n_rows, B.n_rows, arma::fill::zeros);

        for (uint t = 0; t < n_time_steps; ++t) {
            // reset indices and weights
            double time_weight_mean = 0; 
            double time_weight_prec = 0; 
            
            // necessary values to calculate gradients
            uint time_index = sample_info.time_indices(t);
            arma::vec coefs = moments.coefs.col(t);
            
            for (uint i = 0; i < n_nodes; ++i) {
                uint dyad_idx = 0.;
                arma::vec omega_it = omega(t, i);
                arma::vec mu_i = moments.U.tube(i, t);
                double degree = sample_info.degrees(t, i);
                double nonedge_weight = sample_info.weights(t, i);
                for (auto j : sample_info.dyad_indices(t, i)) {
                    //if (j < i) {
                    // get necessary variables to calculate gradients
                    double z = alpha * (Y(time_index)(i, j) - 0.5);
                    
                    arma::vec x = get_covariates(X(time_index), i, j);
                    arma::vec mu_j = moments.U.tube(j, t);
                    
                    // dyad weight
                    double sample_weight = (
                            (dyad_idx < degree) ? 1.0 : nonedge_weight);
                    
                    // gradient of the mean
                    double residual = z - omega_it(dyad_idx) * arma::as_scalar(
                        coefs.t() * x - coefs(k) * x(k) + mu_i.t() * mu_j);

                    time_weight_mean += sample_weight * x(k) * residual; 

                    // gradient of the precision
                    time_weight_prec += (
                            sample_weight * omega_it(dyad_idx) * pow(x(k), 2));

                    //} 
                    dyad_idx += 1;
                }
            }
            
            // XXX: sample connections from a single node
            //uint i = arma::randi(1, arma::distr_param(0, n_nodes-1))(0);
            //uint dyad_idx = 0.;
            //arma::vec omega_it = omega(t, i);
            //arma::vec mu_i = moments.U.tube(i, t);
            //double degree = sample_info.degrees(t, i);
            //double nonedge_weight = sample_info.weights(t, i);
            //for (auto j : sample_info.dyad_indices(t, i)) {
            //    // get necessary variables to calculate gradients
            //    double z = Y(time_index)(i, j) - 0.5;
            //    arma::vec x = X(time_index).tube(i, j);
            //    arma::vec mu_j = moments.U.tube(j, t);
            //    
            //    // dyad weight
            //    double sample_weight = (
            //            (dyad_idx < degree) ? 1.0 : nonedge_weight);
            //    
            //    // gradient of the mean
            //    double residual = z -  omega_it(dyad_idx) * arma::as_scalar(
            //        coefs.t() * x - coefs(k) * x(k) + mu_i.t() * mu_j);

            //    time_weight_mean += sample_weight * x(k) * residual; 

            //    // gradient of the precision
            //    time_weight_prec += (
            //            sample_weight * omega_it(dyad_idx) * pow(x(k), 2));

            //    dyad_idx += 1;
            //
            //}

            grad_mean += time_weight_mean * B.col(t);
            grad_prec += time_weight_prec * (B.col(t) * B.col(t).t());
        }
        
        // re-weight sample for sampling of time points
        double time_weight = (double) Y.n_elem / n_time_steps;
        //grad_mean *= time_weight;
        //grad_prec *= time_weight;
        grad_mean *= 0.5 * time_weight;
        grad_prec *= 0.5 * time_weight;

        // add the prior precision to grad_prec
        grad_prec += prior_precision;
        
        return {grad_mean, grad_prec};
    }
 
}
