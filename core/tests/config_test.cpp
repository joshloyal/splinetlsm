#include <cmath>
#include "asserts.h"
#include <catch2/catch_test_macros.hpp>

#include "splinetlsm.h"
#include "datasets.h"


TEST_CASE("Config", "[config]") {
    
    uint n_nodes = 100;
    int random_state = 123;
    uint n_time_points = 10;
    uint n_covariates = 3;

    auto [Y, B, X, time_points] = splinetlsm::load_dynamic_network(
            n_nodes, n_time_points, n_covariates, random_state);

    
    splinetlsm::ModelConfig config(Y, B, X, 2, 1, 2);
    
    REQUIRE(config.n_nodes == n_nodes);
    REQUIRE(config.n_covariates == n_covariates);
    REQUIRE(config.n_knots == B.n_rows);
    
    std::cout << config.diff_matrix << std::endl;
    std::cout << config.penalty_matrix << std::endl;
    std::cout << config.coefs_diff_matrix << std::endl;
    std::cout << config.coefs_penalty_matrix << std::endl;
}
