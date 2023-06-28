#include <cmath>
#include <utility>

#include "splinetlsm.h"


namespace splinetlsm {
    std::tuple<sp_cube, arma::sp_mat, array4d, arma::vec>
    load_dynamic_network(uint n_nodes, uint n_time_points, uint n_covariates,
            int random_state) {
        arma::arma_rng::set_seed(random_state);
        
        // equally space time points
        arma::vec time_points = arma::linspace(0, 1, n_time_points);

        int n_knots = std::ceil(0.5 * n_time_points);
        //arma::sp_mat B = arma::sprandu<arma::sp_mat>(n_knots, n_time_points, 0.2);
        arma::sp_mat B = arma::speye(n_knots, n_time_points);

        sp_cube Y(n_time_points);
        array4d X(n_time_points);
        for (uint t = 0; t < n_time_points; ++t) {
            // generate independent ER(p = 0.25) networks
            arma::mat y_tilde = arma::randu<arma::mat>(n_nodes, n_nodes);
            arma::mat y = arma::conv_to<arma::mat>::from(y_tilde > 0.9);
            Y(t) = arma::sp_mat(arma::trimatu(y, 1));
            Y(t) += Y(t).t();

            //Y(t) = arma::speye(n_nodes, n_nodes);
            X(t) = arma::cube(n_nodes, n_nodes, n_covariates);
            for (uint k = 0; k < n_covariates; ++k) {
                arma::mat x_tilde = arma::randn<arma::mat>(n_nodes, n_nodes);
                X(t).slice(k) = arma::trimatu(x_tilde, 1);
                X(t).slice(k) += X(t).slice(k).t();
            }
        }

        return {Y, B, X, time_points};
    }
}
