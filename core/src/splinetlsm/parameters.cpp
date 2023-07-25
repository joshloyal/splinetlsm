#include <math.h>
#include "splinetlsm.h"


namespace splinetlsm {
    
    double logit(double x) {
        return log(x / (1 - x));
    }

    NaturalParams::NaturalParams(ModelConfig& config) : 
            W(config.n_nodes, config.n_knots, config.n_features, arma::fill::randn), 
            W_sigma(config.n_features),
            W_coefs(config.n_knots, config.n_covariates, arma::fill::randn),
            W_coefs_sigma(config.n_knots, config.n_knots, config.n_covariates),
            b(config.n_nodes, arma::fill::value(100)),
            b_coefs(config.n_covariates, arma::fill::value(100)),
            mgp_rate(config.n_features, arma::fill::value(100)),
            mgp_shape(config.n_features) {
        
        
        for (uint h = 0; h < config.n_features; ++h) {
            W.slice(h) = arma::cumsum(0.1 * W.slice(h), 1);
        }


        // initialize covariances to the identity
        W_sigma.fill(arma::cube(config.n_knots, config.n_knots, config.n_nodes, 
            arma::fill::zeros));
        for (uint h = 0; h < config.n_features; ++h) {
            for (uint i = 0; i < config.n_nodes; ++i) {
                W_sigma(h).slice(i) = arma::mat(
                        config.n_knots, config.n_knots, arma::fill::eye);
            }
        }
        
        // intercept initialized to the average density of the network
        for (uint k = 0; k < config.n_covariates; ++k) {
            if (k == 0) {
                W_coefs.col(0).fill(logit(arma::mean(config.density)));
            } else {
                W_coefs.col(k) = arma::cumsum(W_coefs.col(k));
            }
        }

        for (uint k = 0; k < config.n_covariates; ++k) {
            W_coefs_sigma.slice(k) = arma::eye(config.n_knots, config.n_knots);
        }


        // GIG parameters
        a = config.rate_prior;
        p = 0.5 * (config.shape_prior - config.n_features * 
            (config.n_knots - config.penalty_order));
        
        a_coefs = config.coefs_rate_prior;
        p_coefs = 0.5 * (config.coefs_shape_prior - 
                (config.n_knots - config.coefs_penalty_order));
        
        mgp_shape(0) = 2 * config.mgp_a1 + config.n_features * config.n_nodes * config.n_knots;
        for (uint h = 1; h < config.n_features; ++h) {
            mgp_shape(h) = (2 * config.mgp_a2 + 
                (config.n_features - h) * config.n_nodes * config.n_knots);
        }

    }
    
    double NaturalParams::operator-(NaturalParams const& params) {
        double res = 0.;
        double n_params = 0.;
        //double rel = 0.;

        res += arma::accu(arma::square(W - params.W));
        //rel += arma::accu(arma::square(params.W));
        n_params += W.n_elem;

        for (uint h = 0; h < W_sigma.n_elem; ++h) {
            res += arma::accu(arma::square(W_sigma(h) - params.W_sigma(h)));
            //rel += arma::accu(arma::square(params.W_sigma(h)));
            n_params += W_sigma(h).n_elem;
        }

        res += arma::accu(arma::square(W_coefs - params.W_coefs));
        //rel += arma::accu(arma::square(params.W_coefs));
        n_params += W_coefs.n_elem;

        res += arma::accu(arma::square(W_coefs_sigma - params.W_coefs_sigma));
        //rel += arma::accu(arma::square(params.W_coefs_sigma));
        n_params += W_coefs_sigma.n_elem;

        res += arma::accu(arma::square(b - params.b));
        //rel += arma::accu(arma::square(params.b));
        n_params += b.n_elem;

        res += arma::accu(arma::square(b_coefs - params.b_coefs));
        //rel += arma::accu(arma::square(params.b_coefs));
        n_params += b_coefs.n_elem;

        res += arma::accu(arma::square(mgp_rate - params.mgp_rate));
        //rel += arma::accu(arma::square(params.mgp_rate));
        n_params += mgp_rate.n_elem;
        
        //return res / rel;
        return res / n_params;
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
                W_sigma(h).slice(i) = arma::inv(
                    natural_params.W_sigma(h).slice(i), arma::inv_opts::allow_approx);
                W.slice(h).row(i) = (W_sigma(h).slice(i) * 
                        natural_params.W.slice(h).row(i).t()).t();
            }
        }

        // coefficient weights
        for (uint k = 0; k < n_covariates; ++k) {
            W_coefs_sigma.slice(k) = arma::inv(
                natural_params.W_coefs_sigma.slice(k), arma::inv_opts::allow_approx);
            W_coefs.col(k) = (W_coefs_sigma.slice(k) * 
                natural_params.W_coefs.col(k));
        }
     
    }

    ModelParams::ModelParams() {}
    
    Params::Params(ModelConfig& config) : 
        natural(config), model(natural) {}
    
    Params::Params(ModelConfig& config, arma::cube& W_init, arma::mat& W_coefs_init) : 
        natural(config), model(natural) {
        
        natural.W = W_init;
        model.W = W_init;

        natural.W_coefs = W_coefs_init;
        model.W_coefs = W_coefs_init;
    }

    Params::Params(NaturalParams& natural, ModelParams& model) :
        natural(natural), model(model) {}

}
