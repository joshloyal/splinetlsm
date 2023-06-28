#pragma once


namespace splinetlsm {

    class NonEdgeSampler {
    public:
        NonEdgeSampler(double proportion);
        std::tuple<arma::field<arma::uvec>, arma::mat, arma::mat> 
            draw(const sp_cube& Y, const arma::uvec& time_indices);
    private:
        double proportion_;
    };

}
