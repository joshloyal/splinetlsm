#pragma once

namespace splinetlsm {
    double roc_auc_score(
        const sp_cube& Y, const array4d& X, const Moments& moments,
        SampleInfo& sample_info);
    
    double log_likelihood(
        const sp_cube& Y, const array4d& X, const Moments& moments,
        SampleInfo& sample_info);
}
