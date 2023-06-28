#pragma once

namespace splinetlsm {

    double calculate_omega(Moments& moments, const arma::cube& X, 
            uint i, uint j, uint t);
    
}
