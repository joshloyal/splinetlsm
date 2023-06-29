#pragma once

namespace splinetlsm {

    double optimize_omega_single(const Moments& moments, const arma::cube& X, 
            uint i, uint j, uint t);

    arma::field<arma::vec> optimize_omega(const Moments& moments,  
            const array4d& X, const SampleInfo& sample_info);
    
}
