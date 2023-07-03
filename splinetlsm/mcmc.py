import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from jax import random, vmap, jit
from jax.scipy.special import expit
from numpyro import handlers
from numpyro.contrib.control_flow import scan
from numpyro.infer import MCMC, NUTS
from sklearn.utils import check_random_state

from .bspline import bspline_basis


MAX_LAMBDA = 1e4


def dynamic_adjacency_to_vec(Y):
    n_time_points, n_nodes, _ = Y.shape
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    subdiag = np.tril_indices(n_nodes, k=-1)
    y = jnp.zeros((n_time_points, n_dyads), dtype=int)
    for t in range(n_time_points):
        y = y.at[t].set(Y[t][subdiag].astype('int32'))

    return y


def splinelsm(Y, B, n_time_points, n_nodes, n_features=10, train_indices=None, 
              is_predictive=False):
    n_knots = B.shape[1]
    
    # intercept
    #sigma_intercept = numpyro.sample("sigma_intercept", dist.HalfCauchy(1))
    #D1 = np.diff(np.eye(n_knots + 1))[1:-1]
    #D2 = np.diff(np.eye(n_knots + 2), n=2)[2:-2]
    #R = (D1.T @ D1 + D2.T @ D2)  / sigma_intercept 
    #R = R.at[0,0].set(R[0,0] + 1)
    #R = R.at[1,1].set(R[1,1] + 1)
    #W_intercept = numpyro.sample("W_intercept", 
    #        dist.MultivariateNormal(precision_matrix=R))

    sigma_intercept = numpyro.sample("sigma_intercept", dist.HalfCauchy(1))
    w0_intercept = numpyro.sample("w0_intercept", dist.Normal(0, 1))
    def transition(y_prev, t):
        y_curr = y_prev + sigma_intercept * numpyro.sample('wt_intercept',
                dist.Normal(0, 1))
        return y_curr, y_curr
    _, wt_intercept = scan(transition, w0_intercept, jnp.arange(1, n_knots))
    W_intercept = jnp.concatenate((jnp.expand_dims(w0_intercept, -1), wt_intercept))
    
    intercept = numpyro.deterministic("intercept", B @ W_intercept)

    # multiplicative-gamma process
    a1 = 2
    a2 = 3
    v1 = numpyro.sample("v1", dist.Gamma(a1, 1))
    vh = numpyro.sample("vh", dist.Gamma(
        jnp.repeat(a2, n_features-1), jnp.ones(n_features - 1)))
    lmbda = numpyro.deterministic("lambda", 
            jnp.cumprod(jnp.concatenate((jnp.array([v1]), vh))))

    # latent position weights
    sigma = numpyro.sample("sigma", dist.Cauchy(jnp.ones(n_nodes))).reshape(-1, 1)
    W0 = numpyro.sample("W0",
            dist.Normal(jnp.zeros((n_nodes, n_features)),
                        jnp.ones((n_nodes, n_features))))
    def transition(y_prev, t):
        y_curr = y_prev + sigma * numpyro.sample('Wt', dist.Normal(
            jnp.zeros((n_nodes, n_features)),
            jnp.ones((n_nodes, n_features))))
        return y_curr, y_curr
    _, Wt = scan(transition, W0, jnp.arange(1, n_knots))
    
    W = numpyro.deterministic("W",
            jnp.concatenate((jnp.expand_dims(W0, axis=0), Wt)))
    X = numpyro.deterministic("X",
            jnp.einsum("tk,knd->tnd", B, W) * (1 / jnp.sqrt(lmbda)))
    
    # calculate likelihood
    subdiag = jnp.tril_indices(n_nodes, k=-1)
    def loglikelihood_fun(carry, t):
        X = carry

        eta = intercept[t] + (X[t] @ X[t].T)[subdiag]

        with numpyro.handlers.mask(mask=train_indices[t]):
            y = numpyro.sample("Y", dist.Bernoulli(logits=eta))

        if is_predictive:
            numpyro.deterministic("probas", expit(eta))
            numpyro.deterministic("logits", eta)
        return X, y

    n_dyads = int(0.5 * n_nodes * (n_nodes-1))
    with numpyro.handlers.condition(data={"Y" : Y}):
        _, ys = scan(loglikelihood_fun, X, jnp.arange(n_time_points))


def predict_proba(model, rng_key, samples, *model_args, **model_kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    model_trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    return model_trace["probas"]["value"]


def predict_logit(model, rng_key, samples, *model_args, **model_kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    model_trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    return model_trace["logits"]["value"]

def posterior_predictive(model, rng_key, samples, stat_fun, *model_args,
                         **model_kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    model_trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    return stat_fun(model_trace["Y"]["value"])


class SplineDynamicLSM(object):
    def __init__(self,
                 n_features=10,
                 n_knots=10,
                 degree=3,
                 random_state=42):
        self.n_features = n_features
        self.n_knots = n_knots
        self.degree = degree
        self.random_state = random_state

    def sample(self, Y, time_points, n_warmup=1000, n_samples=1000, adapt_delta=0.8):
        numpyro.enable_x64()

        n_time_points, n_nodes, _ = Y.shape
        y = dynamic_adjacency_to_vec(Y)
        train_indices = y != -1
        
        self.time_points_ = time_points
        B, self.bs_ = bspline_basis(
                self.time_points_, n_knots=self.n_knots, degree=self.degree,
                return_sparse=False)
        
        model_args = (y, B.T, n_time_points, n_nodes, self.n_features, train_indices)
        model_kwargs = {'is_predictive' : False}
        
        kernel = NUTS(splinelsm, target_accept_prob=adapt_delta)
        self.mcmc_ = MCMC(
            kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=1)

        rng_key = random.PRNGKey(self.random_state)
        self.mcmc_.run(rng_key, *model_args,  **model_kwargs)
        self.samples_ = self.mcmc_.get_samples()
        self.samples_ = jax.tree_map(lambda x : np.array(x), self.samples_)

        return self
    
    @property
    def model_args_(self):
        _, n_time_points, n_nodes, _ = self.samples_['W'].shape
        B = self.bs_(self.time_points_)
        model_args = (None, B, n_time_points, n_nodes, self.n_features,
                      jnp.repeat(True, n_time_points))
        return model_args

    @property
    def model_kwargs_(self):
        return {'is_predictive': True}

    def predict_proba(self, time_points=None, random_state=0):
        rng_key = random.PRNGKey(random_state)
        _, _, n_nodes, _ = self.samples_['W'].shape
        
        # re-calculate Xs from Ws
        if time_points is None:
            samples = {k: v for k, v in self.samples_.items() if k != 'X'}
        else:
            samples = self.samples_

        time_points = self.time_points_ if time_points is None else time_points
        B = self.bs_(time_points)
        n_time_points = B.shape[0]
        model_args = (None, B, n_time_points, n_nodes, self.n_features,
                      jnp.repeat(True, n_time_points))
        
        n_samples  = self.samples_['W'].shape[0]
        vmap_args = (samples, random.split(rng_key, n_samples))
        return vmap(
            lambda samples, rng_key : predict_proba(
                splinelsm, rng_key, samples,
                *model_args, **self.model_kwargs_)
        )(*vmap_args).mean(axis=0)

    def predict_logits(self, time_points=None, random_state=0):
        rng_key = random.PRNGKey(random_state)
        _, _, n_nodes, _ = self.samples_['W'].shape
        
        # re-calculate Xs from Ws
        if time_points is None:
            samples = {k: v for k, v in self.samples_.items() if k != 'X'}
        else:
            samples = self.samples_

        time_points = self.time_points_ if time_points is None else time_points
        B = self.bs_(time_points)
        n_time_points = B.shape[0]
        model_args = (None, B, n_time_points, n_nodes, self.n_features,
                      jnp.repeat(True, n_time_points))
        
        n_samples  = self.samples_['W'].shape[0]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))
        return vmap(
            lambda samples, rng_key : predict_logit(
                splinelsm, rng_key, samples,
                *model_args, **self.model_kwargs_)
        )(*vmap_args).mean(axis=0)
    
    def posterior_predictive(self, stat_fun, time_points=None, random_state=42):
        rng_key = random.PRNGKey(random_state)
        _, _, n_nodes, _ = self.samples_['W'].shape
        
        # re-calculate Xs from Ws
        if time_points is None:
            samples = {k: v for k, v in self.samples_.items() if k != 'X'}
        else:
            samples = self.samples_

        time_points = self.time_points_ if time_points is None else time_points
        B = self.bs_(time_points)
        n_time_points = B.shape[0]
        model_args = (None, B, n_time_points, n_nodes, self.n_features,
                      jnp.repeat(True, n_time_points))
        
        n_samples  = self.samples_['W'].shape[0]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))
        return np.asarray(vmap(
            lambda samples, rng_key : posterior_predictive(
                splinelsm, rng_key, samples, stat_fun,
                *model_args, **self.model_kwargs_)
        )(*vmap_args))
