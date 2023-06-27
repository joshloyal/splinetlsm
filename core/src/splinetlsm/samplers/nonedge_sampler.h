#pragma once


namespace splinetlsm {

    class NonEdgeSampler {
    public:
        NonEdgeSampler(double proportion);
        arma::field<arma::uvec> draw(arma::uvec& time_indices);
    private:
        double proportion_;
    };

}
