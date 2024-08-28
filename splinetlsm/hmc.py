import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from math import ceil
from jax import random, vmap, jit
from jax.scipy.special import expit
from numpyro import handlers
from numpyro.contrib.control_flow import scan
from numpyro.infer import MCMC, NUTS
from sklearn.utils import check_random_state
from sklearn.metrics import roc_auc_score

from .bspline import bspline_basis
from .mcmc_utils import condition


def dynamic_adjacency_to_vec(Y):
    n_time_points, n_nodes, _ = Y.shape
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    subdiag = np.tril_indices(n_nodes, k=-1)
    y = np.zeros((n_time_points, n_dyads))
    for t in range(n_time_points):
        y[t] = Y[t][subdiag]

    return y


def splinelsm(Y, B, X, n_time_points, n_nodes, n_features=10, alpha=1.0, 
              train_indices=None, is_predictive=False):
    n_segments = B.shape[1]   # (n_time_points, n_segments)

    sigma_intercept = numpyro.sample("sigma_intercept", dist.Gamma(1, 0.5))
    W0 = numpyro.sample("W0_intercept", dist.Normal(0, 100)) 
    W_intercept = W0 + jnp.cumsum(
            jnp.sqrt(sigma_intercept) * 
                numpyro.sample("W_intercept", dist.Normal(np.zeros(n_segments-1))))
    W_intercept = jnp.r_[W0, W_intercept].reshape(-1, 1)
    intercept = numpyro.deterministic("intercept", B @ W_intercept)
    
    if X is not None:
        n_covariates = X.shape[-1]
        sigma_coefs = numpyro.sample("sigma_coefs", 
            dist.Gamma(jnp.ones(n_covariates), 0.5 * jnp.ones(n_covariates)))
        W0_coefs = numpyro.sample("W0_coefs", 
                dist.Normal(jnp.zeros(n_covariates), 100 * jnp.ones(n_covariates)))
        W_coefs = W0_coefs + jnp.cumsum(
            jnp.sqrt(sigma_coefs) * numpyro.sample("W_coefs", dist.Normal(
                jnp.zeros((n_segments-1, n_covariates)), jnp.ones((n_segments-1, n_covariates)))),
            axis=0)
        W_coefs = jnp.concatenate((jnp.expand_dims(W0_coefs, axis=0), W_coefs))  # n_segments x n_covariates
        coefs = numpyro.deterministic("coefs", B @ W_coefs)  # shape (n_time_steps, n_covariates)

    # multiplicative-gamma process
    a1 = 2
    a2 = 3
    v1 = numpyro.sample("v1", dist.Gamma(a1, 1))
    vh = numpyro.sample("vh", dist.Gamma(
        jnp.repeat(a2, n_features-1), jnp.ones(n_features - 1)))
    lmbda = numpyro.deterministic("lambda", 
            jnp.cumprod(jnp.concatenate((jnp.array([v1]), vh))))

    # latent position weights
    sigma = numpyro.sample("sigma", dist.Gamma(jnp.ones(n_nodes), 0.5 * jnp.ones(n_nodes))).reshape(-1, 1)
    W0 = numpyro.sample("W0",
            dist.Normal(jnp.zeros((n_nodes, n_features)),
                        jnp.ones((n_nodes, n_features))))
    def transition(y_prev, t):
        y_curr = y_prev + sigma * numpyro.sample('Wt', dist.Normal(
            jnp.zeros((n_nodes, n_features)),
            jnp.ones((n_nodes, n_features))))
        return y_curr, y_curr
    _, Wt = scan(transition, W0, jnp.arange(1, n_segments))
    
    W = numpyro.deterministic("W",
            jnp.concatenate((jnp.expand_dims(W0, axis=0), Wt)))
    U = numpyro.deterministic("U",
            jnp.einsum("tk,knd->tnd", B, W) * (1 / jnp.sqrt(lmbda)))
    
    # calculate likelihood
    subdiag = jnp.tril_indices(n_nodes, k=-1)
    def loglikelihood_fun(carry, t):
        U = carry

        eta = intercept[t] + (U[t] @ U[t].T)[subdiag]
        if X is not None:
            eta += X[t] @ coefs[t]

        with numpyro.handlers.mask(mask=train_indices[t]):
            with numpyro.handlers.scale(scale=alpha):
                y = numpyro.sample("Y", dist.Bernoulli(logits=eta))

        if is_predictive:
            numpyro.deterministic("probas", expit(eta))
            numpyro.deterministic("logits", eta)
        return U, y

    n_dyads = int(0.5 * n_nodes * (n_nodes-1))
    with numpyro.handlers.condition(data={"Y" : Y}):
        _, ys = scan(loglikelihood_fun, U, jnp.arange(n_time_points))


def predict_proba(model, rng_key, samples, *model_args, **model_kwargs):
    model = handlers.seed(condition(model, samples), rng_key)
    model_trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    return model_trace["probas"]["value"]


def posterior_predictive(model, rng_key, samples, stat_fun, *model_args,
                         **model_kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    model_trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    return stat_fun(model_trace["Y"]["value"])


class SplineDynamicLSMHMC(object):
    def __init__(self,
                 n_features=2,
                 n_segments='auto',
                 degree=3,
                 alpha=0.95,
                 random_state=42):
        self.n_features = n_features
        self.n_segments = n_segments
        self.degree = degree
        self.alpha = alpha
        self.random_state = random_state

    def sample(self, Y, time_points, X=None, n_warmup=1000, n_samples=1000, adapt_delta=0.8):
        numpyro.enable_x64()

        n_time_points, n_nodes, _ = Y.shape
        y = jnp.array(dynamic_adjacency_to_vec(Y))
        train_indices = y != -1
        
        # scale time_points to [0, 1] interval
        self.time_min_ = np.min(time_points)
        self.time_max_ = np.max(time_points)
        self.time_points_ = ((time_points - self.time_min_) / 
            (self.time_max_ - self.time_min_))
        
        if self.n_segments == 'auto':
            self.n_segments_ = max(5, min(
                    ceil((n_nodes * n_time_points) ** 0.2) + 1, 36))
        else:
            self.n_segments_ = self.n_segments
        
        if X is not None:
            n_covariates = X.shape[-1]
            n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
            self.X_fit_ = np.zeros((n_time_points, n_dyads, n_covariates))
            for k in range(n_covariates):
                self.X_fit_[..., k] = dynamic_adjacency_to_vec(X[..., k])
            self.X_fit_ = jnp.asarray(self.X_fit_)
        else:
            self.X_fit_ = None

        self.B_fit_ = bspline_basis(
                self.time_points_, n_segments=self.n_segments_, 
                degree=self.degree, return_sparse=False).T
        
        model_args = (y, self.B_fit_, self.X_fit_, n_time_points, n_nodes, 
                self.n_features, self.alpha, train_indices)
        model_kwargs = {'is_predictive' : False}
        
        kernel = NUTS(splinelsm, target_accept_prob=adapt_delta)
        self.mcmc_ = MCMC(
            kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=1)

        rng_key = random.PRNGKey(self.random_state)
        self.mcmc_.run(rng_key, *model_args,  **model_kwargs)
        self.samples_ = self.mcmc_.get_samples()
        self.samples_ = jax.tree_map(lambda x : np.array(x), self.samples_)
        
        self.coefs_ = self.samples_['coefs'].mean(axis=0)
        self.intercept_ = self.samples_['intercept'].mean(axis=0)

        self.probas_ = self.predict_proba()
        self.auc_ = roc_auc_score(y.ravel(), self.probas_.ravel()) 

        return self
    
    @property
    def model_args_(self):
        n_time_points, n_nodes, _ = self.samples_['U'].shape
        model_args = (None, self.B_fit_, self.X_fit_,  n_time_points, n_nodes, self.n_features,
                      self.alpha, jnp.repeat(True, n_time_points))
        return model_args

    @property
    def model_kwargs_(self):
        return {'is_predictive': True}

    def predict_proba(self, time_points=None, random_state=0):
        rng_key = random.PRNGKey(random_state)
        _, n_time_points, n_nodes, _ = self.samples_['U'].shape
        
        model_args = (None, self.B_fit_, self.X_fit_, n_time_points, n_nodes, self.n_features,
                      self.alpha, jnp.repeat(True, n_time_points))
        
        n_samples  = self.samples_['U'].shape[0]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))
        return vmap(
            lambda samples, rng_key : predict_proba(
                splinelsm, rng_key, samples,
                *model_args, **self.model_kwargs_)
        )(*vmap_args).mean(axis=0)
 
    def posterior_predictive(self, stat_fun, random_state=42):
        rng_key = random.PRNGKey(random_state)
        _, n_time_points, n_nodes, _ = self.samples_['U'].shape
        
        model_args = (None, self.B_fit_, self.X_fit_, 
                      n_time_points, n_nodes, self.n_features,
                      self.alpha, jnp.repeat(True, n_time_points))
        
        n_samples = self.samples_['U'].shape[0]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))
        return np.asarray(vmap(
            lambda samples, rng_key : posterior_predictive(
                splinelsm, rng_key, samples, stat_fun,
                *model_args, **self.model_kwargs_)
        )(*vmap_args))
