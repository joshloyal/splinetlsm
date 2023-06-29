#include <cmath>
#include "asserts.h"
#include <catch2/catch_test_macros.hpp>

#include "splinetlsm.h"
#include "datasets.h"


TEST_CASE("Params", "[params]") {
    
    uint n_nodes = 100;
    int random_state = 123;
    uint n_time_points = 10;
    uint n_covariates = 3;
    uint n_features = 2;

    auto [Y, B, X, time_points] = splinetlsm::load_dynamic_network(
            n_nodes, n_time_points, n_covariates, random_state);

    
    splinetlsm::ModelConfig config(Y, B, X, n_features, 1, 2);
    splinetlsm::Params params(config);
    
    REQUIRE(params.natural.W.n_rows == n_nodes);
    REQUIRE(params.natural.W.n_cols == config.n_knots);
    REQUIRE(params.natural.W.n_slices == n_features);    
    REQUIRE(params.natural.a == params.model.a);

    REQUIRE(params.natural - params.natural == 0.);
    
    splinetlsm::Params params2(config);

    REQUIRE(params.natural - params2.natural > 0.);

    splinetlsm::Params params3(params.natural, params.model);
    
    REQUIRE(params.natural - params3.natural == 0.);
}
