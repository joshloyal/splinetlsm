#pragma once

#include "splinetlsm.h"


namespace splinetlsm {
    std::tuple<sp_cube, arma::sp_mat, array4d, arma::vec>
    load_dynamic_network(uint n_nodes, uint n_time_points, uint n_covariates, 
            int random_state);
}
