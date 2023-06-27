#pragma once

namespace splinetlsm {
    // natural parameters of the variational distribution
    class NaturalParams {
        public:
            NaturalParams(ModelConfig& config);

            double operator-(NaturalParams const& params);

        public:
            arma::cube W;
            array4d W_sigma;

            arma::mat W_coefs;
            arma::cube W_coefs_sigma;

            arma::vec b;
            arma::vec b_coefs;

            arma::vec mgp_rate;
            
            // hyperparamters that do not change
            
            // GIG parameters
            double a;
            double p;
            
            double a_coefs;
            double p_coefs;
            
            // MGP shape parameters
            arma::vec mgp_shape;
    };

    // parameters of the variational distribution
    class ModelParams {
        public:
            ModelParams(NaturalParams& natural_params);
        
        public:
            arma::cube W;              // n x L_m x d array of node weights means E[w_{ih,q}]
            array4d W_sigma;           // d x n x L_m x L_m array of node weight covariances Cov(w_{ih})
            
            arma::mat W_coefs;          // p x L_m array of covariate weights E[w_k]
            arma::cube W_coefs_sigma;   // p x L_m x L_m array of covariate weight covariances Cov(w_k)
            
            arma::vec b;                // n-dimensional vector GIG b parameters for node variances sigma_i
            arma::vec b_coefs;          // p-dimensional vector GIG b paramters for covariates sigma_beta_k
            arma::vec mgp_rate;         // d-dimensional vector of rate parameters for the MGP (q(nu_h) = Gamma(a, rate[h]))

            // hyperparamters that do not change
            
            // GIG parameters
            double a;
            double p;
            
            double a_coefs;
            double p_coefs;
            
            // MGP shape parameters
            arma::vec mgp_shape;
    };
    
}
