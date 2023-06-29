#pragma once


namespace splinetlsm {
    
    const uint N_NONEDGES_ISOLATED_NODES = 10;

    class NonEdgeSampler {
    public:
        NonEdgeSampler(double proportion);
        std::tuple<arma::field<arma::uvec>, arma::mat, arma::mat> 
            draw(const sp_cube& Y, const arma::uvec& time_indices);
    private:
        double proportion_;
    };

}
