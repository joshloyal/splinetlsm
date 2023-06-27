#include <vector>
#include <utility>

#include "drforest.h"


namespace splinetlsm {
    
    TimeSampler::TimeSampler(uint n_samples) :
        n_samples_(n_samples) {}

    //std::pair<arma::uvec, arma::uvec>
    arma::uvec TimeSampler::draw(arma::vec& time_points) {
        // uniformely sample time points
        
        uint n_time_steps = time_points.size();
        if (n_samples_ >= n_time_steps) {
            return arma::regspace(0, n_time_steps - 1);
        }
        
        //arma::uvec time_indices = arma::randomperm(n_time_points, n_samples);
        //arma::uvec time_subsample = time_points.elem(time_indices);
        //return { time_subsample, time_indices };
        
        // XXX: sort is for convenience but should we remove it?
        return arma::sort(arma::randomperm(time_points.size(), n_samples));
    }
}
