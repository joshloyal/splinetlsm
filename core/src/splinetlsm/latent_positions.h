#pragma once

namespace splinetlsm {

    std::pair<arma::vec, arma::mat> calculate_latent_position_gradients(
            const sp_cube& Y, const array4d& X, const arma::sp_mat& B, 
            Moments& moments, arma::mat& prior_precision, 
            arma::field<arma::vec>& omega, double alpha,
            SampleInfo& sample_info, uint i, uint h);
}
