#include "splinelsm.h"


namespace splinetlsm {
    
    NaturalParams::NaturalParams(ModelConfig& config) : 
            W(config.n_nodes, config.n_knots, config.n_features, arma::fill::randn), 
            W_sigma(config.n_features, arma::fill::ones),
            W_coefs(config.n_covariates, config.n_knots, arma::fill::randn),
            W_coefs_sigma(config.n_covariates, config.n_knots, config.n_knots, arma::fill::ones),
            b(config.n_nodes, arma::fill::ones),
            b_coefs(config.n_covariates, arma::fill:ones),
            mgp_rate(config.n_features, arma::fill::ones),
            mgp_shape(config.n_features) {
        
        //for (uint h = 0; h < config.n_features; ++h) {
        //    W_sigma(h) = arma::cube(config.n_nodes, config.n_knots, config.n_knots);
        //}
        W_sigma.fill(arma::cube(config.n_nodes, config.n_knots, config.n_knots, arma::fill:randn));

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
    
    NaturalParams::operator-(NaturalParams const& params) {
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
            W_sigma(natural_params.n_elem),
            W_coefs(arma::size(natural_params.W_coefs)),
            W_coefs_sigma(arma::size(natural_params.W_coefs_sigma)),
            a(natural_params.a),
            b(natural_params.b),
            p(natural_params.p)
            a_coefs(natural_params.a_coefs),
            b_coefs(natural_params.b_coefs),
            p_coefs(natural_params.p_coefs),
            mgp_rate(natural_params.mgp_rate),
            mgp_shape(natural_params.mgp_shape) {
        
        // node weights
        W_sigma.fill(arma::size(natural_params.W(0)));
        for (uint i = 0; i < n_nodes; ++i) {
            for (uint h = 0; h < n_features; ++h) {
                W_sigma(d).row(i) = inv_sympd(natural_params.W_sigma(d).row(i));
                W(i).col(h) = W_sigma(h).row(i) * nautral_params.W(i).col(h);
            }
        }

        // coefficient weights
        for (uint k = 0; k < n_covariates; ++k) {
            W_coefs_sigma.row(k) = inv_sympd(natural_params.W_coefs_sigma.row(k));
            W_coefs.row(k) = W_coefs_sigma.row(k) * nautral_params.W_coefs.row(k).t();
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

}
