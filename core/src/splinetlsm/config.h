#pragma once

namespace splinetlsm {

    class ModelConfig {
    public:
        ModelConfig(const sp_cube& Y, const sp_mat& B, const array4d& X, 
            uint n_features=2, 
            uint penalty_order=1, uint coefs_penalty_order=1, 
            double rate_prior=2, double shape_prior=1,
            double coefs_rate_prior=2, double coefs_shape_prior=1,
            double mgp_a1=2, double mgp_a2=3,
            double tau_prec=0.01, double coefs_tau_prec=0.01);

    public:    
        uint n_nodes;
        uint n_features;
        uint n_covariates;
        uint n_knots;
        uint penalty_order;
        uint coefs_penalty_order;

        double rate_prior;
        double shape_prior;

        double coefs_rate_prior;
        double coefs_shape_prior;

        double mgp_a1;
        double mgp_a2;

        double tau_prec;
        double coefs_tau_prec;

        arma::mat diff_matrix;
        arma::mat penalty_matrix;
        
        arma::mat coefs_penalty_matrix;
        arma::mat coefs_diff_matrix;
    };

}
