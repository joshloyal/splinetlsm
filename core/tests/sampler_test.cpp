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
    double proportion = 4.;

    auto [Y, B, X, time_points] = splinetlsm::load_dynamic_network(
            n_nodes, n_time_points, n_covariates, random_state);
    
    // include a fully-connected node
    Y(0).col(0) = arma::vec(n_nodes, arma::fill::ones);
    Y(0).row(0) = arma::rowvec(n_nodes, arma::fill::ones);
    Y(0)(0, 0) = 0.;  

    splinetlsm::DyadSampler dyad_sampler(proportion, n_time_samples);
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
            } else if ((proportion * d) >= (n_nodes - 1 - d)) {
                // all non-edges (isolated node)
                weights(i) = 1.;
            } else {
                weights(i) = (double) (n_nodes - 1 - d) / (proportion * d);
            }
        }
        arma::vec calc_weights = sample_info.weights.row(t).t();
        testing::assert_vector_equal(weights, calc_weights);
    }
    

    
    // check dyad list has the correct shape and does not include self loops
    arma::uvec self_loop = arma::find(sample_info.dyad_indices(1, 2) == 2); 
    REQUIRE(sample_info.dyad_indices.n_rows == n_time_samples);
    REQUIRE(sample_info.dyad_indices.n_cols == n_nodes);
    REQUIRE(self_loop.n_elem == 0);
    REQUIRE(sample_info.dyad_indices(1, 2).n_elem == sample_info.degrees(1, 2) * (proportion + 1));

}

TEST_CASE("Sampler Isolated", "[sampler]") {
    uint n_nodes = 100;
    int random_state = 123;
    uint n_time_points = 10;
    uint n_covariates = 3;
    uint n_time_samples = 10;
    double proportion = 4;

    auto [Y, B, X, time_points] = splinetlsm::load_dynamic_network(
            n_nodes, n_time_points, n_covariates, random_state);
    
    // include an isolated node 
    Y(0).row(1) = arma::rowvec(n_nodes, arma::fill::zeros);
    Y(0).col(1) = arma::vec(n_nodes, arma::fill::zeros);
    Y(0)(1, 1) = 0.;
    

    splinetlsm::DyadSampler dyad_sampler(proportion, n_time_samples);
    splinetlsm::SampleInfo sample_info = dyad_sampler.draw(Y, time_points);
    
    REQUIRE(sample_info.time_indices.n_elem == n_time_samples);
    
    // check that the degrees and weights match
    for (uint t = 0; t < sample_info.time_indices.n_elem; ++t) {
        arma::mat D = arma::mat(arma::sum(Y(sample_info.time_indices(t))));
        arma::vec degree = D.row(0).t();
        arma::vec calc_degree = sample_info.degrees.row(t).t();
        testing::assert_vector_equal(degree, calc_degree);
        
        arma::vec weights(n_nodes);
        uint max_degree = n_nodes - 1;
        for (uint i = 0; i < n_nodes; ++i) {
            uint d = degree(i);
            uint n_nonedges = max_degree - d;
            if (d == max_degree) {
                // no non-edges
                weights(i) = 0.;
            } else if (n_nonedges <= std::floor(proportion * d)) {
                weights(i) = 1.;
            } else if (d == 0) {
                // isolated node
                weights(i) = max_degree / 10.;
            } else {
                weights(i) = (double) (n_nodes - 1 - d) / (proportion * d);
            }
        }
        arma::vec calc_weights = sample_info.weights.row(t).t();
        testing::assert_vector_equal(weights, calc_weights);
    }
    
    arma::uvec self_loop = arma::find(sample_info.dyad_indices(1, 2) == 2); 
    REQUIRE(sample_info.dyad_indices.n_rows == n_time_samples);
    REQUIRE(sample_info.dyad_indices.n_cols == n_nodes);
    REQUIRE(self_loop.n_elem == 0);
    REQUIRE(sample_info.dyad_indices(1, 2).n_elem == sample_info.degrees(1, 2) * (proportion + 1));
}
