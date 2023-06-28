#pragma once

namespace splinetlsm {

    double calculate_omega(Moments& moments, const array4d& X, 
            uint i, uint j, uint t, SampleInfo& sample_info);
    
}
