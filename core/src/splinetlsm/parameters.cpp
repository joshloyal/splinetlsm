#include "splinetlsm.h"


namespace splinetlsm {
    
    NaturalParams::NaturalParams(ModelConfig& config) : 
            W(config.n_nodes, config.n_knots, config.n_features, arma::fill::randn), 
            W_sigma(config.n_features),
            W_coefs(config.n_knots, config.n_covariates, arma::fill::randn),
            W_coefs_sigma(config.n_knots, config.n_knots, config.n_covariates,  arma::fill::ones),
            b(config.n_nodes, arma::fill::ones),
            b_coefs(config.n_covariates, arma::fill::ones),
            mgp_rate(config.n_features, arma::fill::ones),
            mgp_shape(config.n_features) {
        
        //for (uint h = 0; h < config.n_features; ++h) {
        //    W_sigma(h) = arma::cube(config.n_nodes, config.n_knots, config.n_knots);
        //}
        W_sigma.fill(arma::cube(config.n_knots, config.n_knots, config.n_nodes, arma::fill::randn));

        // GIG parameters
        a = config.rate_prior;
        p = 0.5 * (config.shape_prior - config.n_features * 
            (config.n_knots - config.penalty_order));
        
        a_coefs = config.coefs_rate_prior;
        p_coefs = 0.5 * (config.rate_prior - 
                (config.n_knots - config.coefs_penalty_order));
        
        mgp_shape(0) = 2 * config.mgp_a1 + config.n_features * config.n_nodes * config.n_knots;
        for (uint h = 1; h < config.n_features; ++h) {
            mgp_shape(h) = (2 * config.mgp_a2 + 
                (config.n_features - h + 1) * config.n_nodes * config.n_knots);
        }

    }
    
    double NaturalParams::operator-(NaturalParams const& params) {
        double res = 0.;

        res += arma::accu(arma::abs(W - params.W));
        for (uint h = 0; h < W_sigma.n_elem; ++h) {
            res += arma::accu(arma::abs(W_sigma(h) - params.W_sigma(h)));
        }
        res += arma::accu(arma::abs(W_coefs - params.W_coefs));
        res += arma::accu(arma::abs(W_coefs_sigma - params.W_coefs_sigma));
        res += arma::accu(arma::abs(b - params.b));
        res += arma::accu(arma::abs(b_coefs - params.b_coefs));
        res += arma::accu(arma::abs(mgp_rate - params.mgp_rate));
        
        return res;
    }


    ModelParams::ModelParams(NaturalParams& natural_params) : 
            W(arma::size(natural_params.W)), 
            W_sigma(natural_params.W_sigma.n_elem),
            W_coefs(arma::size(natural_params.W_coefs)),
            W_coefs_sigma(arma::size(natural_params.W_coefs_sigma)),
            b(natural_params.b),
            b_coefs(natural_params.b_coefs),
            mgp_rate(natural_params.mgp_rate),
            a(natural_params.a),
            p(natural_params.p),
            a_coefs(natural_params.a_coefs),
            p_coefs(natural_params.p_coefs),
            mgp_shape(natural_params.mgp_shape) {
        
        uint n_nodes = W.n_rows;
        uint n_features = W.n_slices;
        uint n_covariates = W_coefs.n_cols;

        // node weights
        W_sigma.fill(arma::cube(arma::size(natural_params.W_sigma(0))));
        for (uint i = 0; i < n_nodes; ++i) {
            for (uint h = 0; h < n_features; ++h) {
                W_sigma(h).slice(i) = inv_sympd(natural_params.W_sigma(h).slice(i));
                W.slice(h).row(i) = W_sigma(h).slice(i) * natural_params.W.slice(h).row(i);
            }
        }

        // coefficient weights
        for (uint k = 0; k < n_covariates; ++k) {
            W_coefs_sigma.slice(k) = inv_sympd(natural_params.W_coefs_sigma.slice(k));
            W_coefs.col(k) = W_coefs_sigma.slice(k) * natural_params.W_coefs.col(k);
        }
        
        // variance parameters
        //a = natural_parameters.a;
        //b = natural_params.b;
        //p = natural_params.p;

        //a_coefs = natural_params.a_coefs;
        //b_coefs = natural_params.b_coefs;
        //p_coefs = natural_params.p_coefs;
        
        //mgp_shape = natural_params.mgp_shape;
        //mgp_rate = natural_params.mgp_rate;
        
    }
    
    Params::Params(ModelConfig& config) : 
        natural(config), model(natural) {}

    Params::Params(NaturalParams& natural, ModelParams& model) :
        natural(natural), model(model) {}

}
