#include <cmath>

#include "splinetlsm.h"
#include "AUROC.h"

namespace splinetlsm {
    double roc_auc_score(
        const sp_cube& Y, const array4d& X, const Moments& moments,
        SampleInfo& sample_info) {
        
        uint n_nodes = Y(0).n_rows;
        uint n_time_steps = sample_info.time_indices.n_elem;

        std::vector<uint> y_true;
        std::vector<double> y_score;

        for (uint t = 0; t < n_time_steps; ++t) {
            // necessary values to calculate logits
            uint time_index = sample_info.time_indices(t);
            arma::vec coefs = moments.coefs.col(t);
            for (uint i = 0; i < n_nodes; ++i) {
                uint dyad_idx = 0.;
                arma::vec mu_i = moments.U.tube(i, t);
                for (auto j : sample_info.dyad_indices(t, i)) {
                    if (i < j) {
                        y_true.push_back(Y(time_index)(i, j));

                        arma::vec x = get_covariates(X(time_index), i, j);
                        arma::vec mu_j = moments.U.tube(j, t);
                        
                        
                        // gradient of the mean
                        y_score.push_back(
                            arma::as_scalar(coefs.t() * x + mu_i.t() * mu_j));
                    }
                    dyad_idx += 1;
                }
            }
        }

        return AUROC(&y_true[0], &y_score[0], y_true.size());
    }

    double log_likelihood(
        const sp_cube& Y, const array4d& X, const Moments& moments,
        SampleInfo& sample_info) {
        
        uint n_nodes = Y(0).n_rows;
        uint n_time_steps = sample_info.time_indices.n_elem;
        
        uint n_samples = 0.;
        double eta = 0.;
        double loglik = 0.;
        for (uint t = 0; t < n_time_steps; ++t) {
            // necessary values to calculate logits
            uint time_index = sample_info.time_indices(t);
            arma::vec coefs = moments.coefs.col(t);
            for (uint i = 0; i < n_nodes; ++i) {
                uint dyad_idx = 0.;
                arma::vec mu_i = moments.U.tube(i, t);
                for (auto j : sample_info.dyad_indices(t, i)) {
                    if (i < j) {
                        arma::vec x = get_covariates(X(time_index), i, j);
                        arma::vec mu_j = moments.U.tube(j, t);
                        
                        // logits 
                        eta = arma::as_scalar(coefs.t() * x + mu_i.t() * mu_j);
                        loglik += Y(time_index)(i, j) * eta - log(1 + exp(eta));
                        n_samples += 1;
                    }
                    dyad_idx += 1;
                }
            }
        }

        return loglik / n_samples;
    }
}
