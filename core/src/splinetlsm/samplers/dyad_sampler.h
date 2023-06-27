#pragma once


namespace splinetlsm {
    
    struct SampleInfo {
        arma::uvec time_indices;
        arma::field<arma::uvec> dyad_indices;
        arma::field<arma::uvec> degrees;
        arma::field<arma::vec> weights;
    };


    class DyadSampler {
    public:
        DyadSampler(double nonedge_proportion, uint n_time_steps);
        SampleInfo draw(const arma::sp_mat& Y, arma::uvec& time_points);
    private:
        double nonedge_proportion_;
        double n_time_steps_;

        NonEdgeSampler nonedge_sampler_;
        TimeSampler time_sampler_;
    };

}
