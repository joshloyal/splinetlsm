#include <cmath>
#include <algorithm>
#include <random>
#include <iterator>

#include "asserts.h"
#include <catch2/catch_test_macros.hpp>

#include "splinetlsm.h"
#include "datasets.h"


TEST_CASE("Sampler", "[sampler]") {
    
    uint n_nodes = 100;
    int random_state = 123;
    uint n_time_points = 10;
    uint n_covariates = 3;
    uint n_time_samples = 10;

    auto [Y, B, X, time_points] = splinetlsm::load_dynamic_network(
            n_nodes, n_time_points, n_covariates, random_state);
    
    Y(0).col(0) = arma::vec(n_nodes, arma::fill::ones);
    Y(0).row(0) = arma::rowvec(n_nodes, arma::fill::ones);
    Y(0)(0, 0) = 0.;
    
    //Y(0).row(1) = arma::rowvec(n_nodes, arma::fill::zeros);
    //Y(0).col(1) = arma::vec(n_nodes, arma::fill::zeros);
    //Y(0)(1, 1) = 0.;
    

    splinetlsm::DyadSampler dyad_sampler(4, n_time_samples);
    splinetlsm::SampleInfo sample_info = dyad_sampler.draw(Y, time_points);
    
    REQUIRE(sample_info.time_indices.n_elem == n_time_samples);
    
    // check that the degrees and weights match
    for (uint t = 0; t < sample_info.time_indices.n_elem; ++t) {
        arma::mat D = arma::mat(arma::sum(Y(sample_info.time_indices(t))));
        arma::vec degree = D.row(0).t();
        arma::vec calc_degree = sample_info.degrees.row(t).t();
        testing::assert_vector_equal(degree, calc_degree);
        
        arma::vec weights(n_nodes);
        for (uint i = 0; i < n_nodes; ++i) {
            uint d = degree(i);
            if (d == n_nodes - 1) {
                // no non-edges
                weights(i) = 0.;
            } else if ((4 * d) >= (n_nodes - 1 - d)) {
                // all non-edges (isolated node)
                weights(i) = 1.;
            } else {
                weights(i) = (double) (n_nodes - 1 - d) / (4 * d);
            }
        }
        arma::vec calc_weights = sample_info.weights.row(t).t();
        testing::assert_vector_equal(weights, calc_weights);
    }
    
    REQUIRE(sample_info.dyad_indices.n_rows == n_time_samples);
    REQUIRE(sample_info.dyad_indices.n_cols == n_nodes);

    std::cout << sample_info.dyad_indices(2, 0).t() << std::endl;
}
