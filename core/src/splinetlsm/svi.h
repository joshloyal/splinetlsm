#pragma once


namespace splinetlsm {
    
    const uint MIN_ITER = 40;
    const uint WINDOW_SIZE = 20;

    struct SVIResult {
        ModelParams params;
        bool converged;
        arma::vec parameter_difference;
        arma::vec insample_auc;
        uint n_iter;
    };

    class SVI {
    public:
        SVI(ModelConfig& config, double nonedge_proporition=5, 
                uint n_time_steps=10, double step_size_delay=1, 
                double step_size_power=0.75);
        
        std::pair<Params, double>
        update(const sp_cube& Y, const arma::sp_mat& B, const array4d& X, 
            const arma::vec& time_points, Params& params);
        
    private:
        ModelConfig config_;
        DyadSampler dyad_sampler_;
        uint iter_idx_;
        double step_size_delay_;
        double step_size_power_;
    };
    
    SVIResult optimize_elbo(
            const sp_cube& Y, const arma::sp_mat& B, const array4d& X,
            const arma::vec& time_points, uint n_features=2,
            uint penalty_order=1, uint coefs_penalty_order=2, 
            double rate_prior=2, double shape_prior=1,
            double coefs_rate_prior=2, double coefs_shape_prior=1,
            double mgp_a1=2, double mgp_a2=3,
            double tau_prec=0.01, double coefs_tau_prec=0.01,
            double nonedge_proportion=5, uint n_time_steps=5,
            double step_size_delay=1, double step_size_power=0.75,
            uint max_iter=100, double tol=0.001, int random_state=42);
        
}
