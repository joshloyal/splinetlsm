#pragma once


namespace splinetlsm {
    class SVI {
    public:
        SVI(ModelConfig& config, double step_size_delay=1, double step_size_power=0.75, double tol=0.001);
       
        std::pair<NaturalParams, ModelParams> 
        update(const sp_cube& Y, const sp_mat& B, const array4d& X,
            arma::uvec& time_points, NaturalParams& natural_params, 
            ModelParams& params);
    
    public:
        bool converged
        
    private:
        ModelConfig config_;
        uint iter_idx_;
        double step_size_delay_;
        double step_size_power_;
        double tol_;
    };
    
    ModelParams optimize_elbo(
            const sp_cube& Y, const sp_mat& B, const array4d& X,
            arma::uvec& time_points, uint n_features=2,
            uint penalty_order=1, uint coefs_penalty_order=1, 
            double rate_prior=2, double shape_prior=1,
            double coefs_rate_prior=2, double coefs_shape_prior=1,
            double mgp_a1=2, double mgp_a2=3,
            double tau_prec=0.01, double tau_coefs_prec=0.01,
            double step_size_delay=1, double step_size_power=0.75,
            uint max_iter=100, int random_state=42);
        
}
