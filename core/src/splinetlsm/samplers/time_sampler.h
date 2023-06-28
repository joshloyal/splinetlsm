#pragma once


namespace splinetlsm {
    class TimeSampler {
    public:
        TimeSampler(uint n_samples);
        arma::uvec draw(const arma::vec& time_points);

    private:
        uint n_samples_;
    };
}
