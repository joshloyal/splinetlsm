#pragma once

namespace splinetlsm {
    
    arma::vec get_covariates(const arma::cube& X, uint i, uint j);

    std::pair<arma::vec, arma::mat> calculate_coef_gradients(
            const sp_cube& Y, const array4d& X, const arma::sp_mat& B, 
            Moments& moments, arma::mat& prior_precision, 
            arma::field<arma::vec>& omega, 
            SampleInfo& sample_info, uint k);

}
