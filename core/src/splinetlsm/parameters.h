#pragma once

namespace splinetlsm {

    // natural parameters of the variational distribution
    class NaturalParams {
    public:
        NaturalParams(ModelConfig& config);

        double operator-(NaturalParams const& params);

    public:
        arma::cube W;                // n x L_m x d array of node weights natural parameters lambda_ih
        array4d W_sigma;             // d x L_m x L_m x n array of node weight precisions Lambda_ih
        
        arma::mat W_coefs;           // L_m x p array of covariate weights natural parameters lambda_k
        arma::cube W_coefs_sigma;    // L_m x L_m x p array of covariate weight precisions Lambda_k
        
        arma::vec b;                 // n-dimensional vector GIG b parameters for node variances sigma_i
        arma::vec b_coefs;           // p-dimensional vector GIG b paramters for covariates sigma_beta_k
        arma::vec mgp_rate;          // d-dimensional vector of rate parameters for the MGP (q(nu_h) = Gamma(a, rate[h]))
        
        // hyperparamters that do not change
        
        // GIG(a, b, p) parameters
        double a;
        double p;
        
        double a_coefs;
        double p_coefs;
        
        // MGP shape parameters
        arma::vec mgp_shape;        // d-dimensional vector of MGP shape parameters
    };

    // parameters of the variational distribution
    class ModelParams {
    public:
        ModelParams();
        ModelParams(NaturalParams& natural_params);
    
    public:
        arma::cube W;                 // n x L_m x d array of node weights means E[w_{ih,q}]
        array4d W_sigma;              // d x L_m x L_m x n array of node weight covariances Cov(w_{ih})
        
        arma::mat W_coefs;            // L_m x p array of covariate weights E[w_k]
        arma::cube W_coefs_sigma;     // L_m x L_m x p array of covariate weight covariances Cov(w_k)
        
        arma::vec b;                  // n-dimensional vector GIG b parameters for node variances sigma_i
        arma::vec b_coefs;            // p-dimensional vector GIG b paramters for covariates sigma_beta_k
        arma::vec mgp_rate;           // d-dimensional vector of rate parameters for the MGP (q(nu_h) = Gamma(a, rate[h]))

        // hyperparamters that do not change
        
        // GIG parameters
        double a;
        double p;
        
        double a_coefs;
        double p_coefs;
        
        // MGP shape parameters
        arma::vec mgp_shape;
    };
    
    class Params {
    public:
        Params(ModelConfig& config);
        Params(NaturalParams& natural, ModelParams& model);

    public:
        NaturalParams natural;
        ModelParams model;
    };
    
}
