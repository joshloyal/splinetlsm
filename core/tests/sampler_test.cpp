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

    auto [Y, B, X, time_points] = splinetlsm::load_dynamic_network(
            n_nodes, n_time_points, n_covariates, random_state);
    
    std::cout << "hi" << std::endl;

    splinetlsm::DyadSampler dyad_sampler(4, 3);
    splinetlsm::SampleInfo sample_info = dyad_sampler.draw(Y, time_points);
    
    std::cout << sample_info.time_indices.t() << std::endl;
    std::cout << sample_info.dyad_indices(0, 0).t() << std::endl;
    std::cout << sample_info.degrees(0, 0) << std::endl;
    std::cout << sample_info.weights(0, 0) << std::endl;

    
    //arma::vec a = arma::shuffle(arma::regspace(0, 10));
    //arma::vec a = arma::randperm<arma::vec>(10, 3);
    //std::cout << a << std::endl;
    
    //arma::vec out;
    //arma::vec in = arma::regspace(0, 10);
    //std::sample(in.begin(), in.end(), std::back_inserter(out), 4,
    //            std::mt19937 {std::random_device{}()});
}
