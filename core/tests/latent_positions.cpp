#include <cmath>
#include "asserts.h"
#include <catch2/catch_test_macros.hpp>

#include "splinetlsm.h"
#include "datasets.h"


TEST_CASE("Latent Positions", "[latent_positions]") {
    
    uint n_nodes = 100;
    int random_state = 123;
    uint n_time_points = 10;
    uint n_covariates = 3;
    uint n_features = 2;
    uint n_time_steps = 5;
    double proportion = 3;

    auto [Y, B, X, time_points] = splinetlsm::load_dynamic_network(
            n_nodes, n_time_points, n_covariates, random_state);

    
    splinetlsm::ModelConfig config(Y, B, X, n_features, 1, 2);
    splinetlsm::Params params(config);
    
    // sub sample b-spline basis
    splinetlsm::DyadSampler dyad_sampler(proportion, n_time_steps);
    splinetlsm::SampleInfo sample_info = dyad_sampler.draw(Y, time_points); 
    arma::sp_mat B_sub = B.cols(sample_info.time_indices);
    
    // get omegas
    splinetlsm::Moments moments = splinetlsm::calculate_moments(
            params.model, B_sub); 
    arma::field<arma::vec> omega = splinetlsm::optimize_omega(
            moments, X, sample_info);

    // calculate latent position gradients
    arma::mat prior_precision = splinetlsm::calculate_prior_precision(
            moments.w_prec(0), config);

    auto [grad_mean, grad_prec] = calculate_latent_position_gradients(
            Y, X, B_sub, moments, prior_precision, omega, sample_info, 0, 1);
    
    uint n_knots = B.n_rows;
    REQUIRE(grad_mean.n_rows == n_knots);
    REQUIRE(grad_prec.n_rows == n_knots);
    REQUIRE(grad_prec.n_cols == n_knots);

}
