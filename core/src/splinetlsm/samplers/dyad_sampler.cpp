#include <vector>

#include "splinetlsm.h"


namespace splinetlsm {
    
    DyadSampler::DyadSampler(double nonedge_proportion, uint n_time_steps) :
        time_sampler_(n_time_steps),
        nonedge_sampler_(nonedge_proportion) {}

    SampleInfo DyadSampler::draw(const sp_cube& Y, const arma::vec& time_points) {

        // sample time points
        arma::uvec time_indices = time_sampler_.draw(time_points);
        
        // sample dyads
        auto [dyad_indices, degrees, weights] = nonedge_sampler_.draw(
                Y, time_indices);
        
        // create sample info struct
        SampleInfo sample_info;
        sample_info.time_indices = time_indices;
        sample_info.dyad_indices = dyad_indices;
        sample_info.degrees = degrees;
        sample_info.weights = weights;

        return sample_info;
    }
}
