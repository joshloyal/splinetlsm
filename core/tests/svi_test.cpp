#include <cmath>
#include "asserts.h"
#include <catch2/catch_test_macros.hpp>

#include "splinetlsm.h"
#include "datasets.h"


TEST_CASE("SVI Init", "[svi]") {
    
    uint n_nodes = 100;
    int random_state = 123;
    uint n_time_points = 10;
    uint n_covariates = 3;
    uint n_features = 2;

    auto [Y, B, X, time_points] = splinetlsm::load_dynamic_network(
            n_nodes, n_time_points, n_covariates, random_state);
    
    splinetlsm::ModelConfig config(Y, B, X, n_features, 1, 2);
    splinetlsm::Params params(config);
    
    splinetlsm::SVI svi(config, 3, 5); 

}


TEST_CASE("SVI Update", "[svi]") {
    
    uint n_nodes = 100;
    int random_state = 123;
    uint n_time_points = 10;
    uint n_covariates = 3;
    uint n_features = 2;

    auto [Y, B, X, time_points] = splinetlsm::load_dynamic_network(
            n_nodes, n_time_points, n_covariates, random_state);
    
    splinetlsm::ModelConfig config(Y, B, X, n_features, 1, 2);
    
    splinetlsm::SVI svi(config, 5, 10); 
    
    splinetlsm::Params params(config);

    svi.update(Y, B, X, time_points, params);
}


TEST_CASE("Optimize ELBO", "[svi]") {
    uint n_nodes = 100;
    int random_state = 123;
    uint n_time_points = 10;
    uint n_covariates = 3;
    uint n_features = 2;

    auto [Y, B, X, time_points] = splinetlsm::load_dynamic_network(
            n_nodes, n_time_points, n_covariates, random_state);

    //splinetlsm::SVIResult result = splinetlsm::optimize_elbo(
    //    Y, B, X, time_points, 0.95);
 
}
