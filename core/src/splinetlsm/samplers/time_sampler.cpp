#include <vector>
#include <utility>

#include "splinetlsm.h"


namespace splinetlsm {
    
    TimeSampler::TimeSampler(uint n_samples) :
        n_samples_(n_samples) {}

    //std::pair<arma::uvec, arma::uvec>
    arma::uvec TimeSampler::draw(const arma::vec& time_points) {
        // uniformely sample time points
        
        uint n_time_steps = time_points.size();
        if (n_samples_ >= n_time_steps) {
            return arma::regspace<arma::uvec>(0, n_time_steps - 1);
        }
        
        //arma::uvec time_indices = arma::randomperm(n_time_points, n_samples);
        //arma::uvec time_subsample = time_points.elem(time_indices);
        //return { time_subsample, time_indices };
        
        // XXX: sort is for convenience but should we remove it?
        return arma::sort(arma::randperm(time_points.size(), n_samples_));
    }
}
