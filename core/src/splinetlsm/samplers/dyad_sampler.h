#pragma once


namespace splinetlsm {
    
    struct SampleInfo {
        arma::uvec time_indices;
        arma::field<arma::uvec> dyad_indices;
        arma::mat degrees;
        arma::mat weights;
    };


    class DyadSampler {
    public:
        DyadSampler(double nonedge_proportion, uint n_time_steps);
        SampleInfo draw(const sp_cube& Y, const arma::vec& time_points);

    private:
        TimeSampler time_sampler_;
        NonEdgeSampler nonedge_sampler_;
    };

}
