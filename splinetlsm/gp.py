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
from sklearn.metrics import roc_auc_score


def dynamic_adjacency_to_vec(Y):
    n_time_points, n_nodes, _ = Y.shape
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    subdiag = np.tril_indices(n_nodes, k=-1)
    y = jnp.zeros((n_time_points, n_dyads), dtype=int)
    for t in range(n_time_points):
        y = y.at[t].set(Y[t][subdiag].astype('int32'))

    return y


def calculate_auc(Y, probas):
    Y = np.asarray(Y)
    n_time_steps, _ = Y.shape
    y_true, y_pred = [], []
    for t in range(n_time_steps):
        y_true.append(Y[t])
    
    return roc_auc_score(np.concatenate(y_true), probas.ravel())


# squared exponential kernel 
def kernel(X, Z, length_scale=0.1):
    dist_sq = jnp.power((X[:, None] - Z) / length_scale, 2.0)
    return jnp.exp(-0.5 * dist_sq)


def gplsm(Y, time_points, n_time_points, n_nodes, n_features=10, length_scale=0.1, 
          train_indices=None, is_predictive=False):
    # multiplicative-gamma process
    a1, a2 = 2, 3
    v1 = numpyro.sample("v1", dist.Gamma(a1, 1))
    vh = numpyro.sample("vh", dist.Gamma(
        jnp.repeat(a2, n_features-1), jnp.ones(n_features - 1)))
    gamma = numpyro.deterministic("gamma", 
            jnp.cumprod(jnp.concatenate((jnp.array([v1]), vh))))
    
    # squared exponential kernel
    cov_t = kernel(time_points, time_points, length_scale=length_scale)

    # intercept
    intercept = numpyro.sample("intercept", dist.MultivariateNormal(
        loc=jnp.zeros(n_time_points), covariance_matrix=cov_t))
    
    # latent positions
    Z = numpyro.sample("Z", dist.MultivariateNormal(
        loc=jnp.zeros((n_nodes, n_features, n_time_points)), covariance_matrix=cov_t))
    Z = Z.transpose((2, 0, 1))
    X = numpyro.deterministic("X", (1. / jnp.sqrt(gamma)) * Z)
 
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


class GaussianProcessDynamicLSM(object):
    def __init__(self,
                 n_features=10,
                 length_scale=0.1,
                 random_state=42):
        self.n_features = n_features
        self.length_scale = length_scale
        self.random_state = random_state

    def sample(self, Y, time_points, n_warmup=1000, n_samples=1000, adapt_delta=0.8):
        numpyro.enable_x64()
        
        if isinstance(Y, list):
            # assumed a list of csc matrices
            Y = np.concatenate([y.toarray() for y in Y])

        n_time_points, n_nodes, _ = Y.shape
        self.Y_fit_ = dynamic_adjacency_to_vec(Y)
        train_indices = self.Y_fit_ != -1
        
        self.time_min_ = np.min(time_points)
        self.time_max_ = np.max(time_points)
        self.time_points_ = ((time_points - self.time_min_) / 
            (self.time_max_ - self.time_min_))
        
        model_args = (self.Y_fit_, self.time_points_, n_time_points, n_nodes, 
                self.n_features, self.length_scale, train_indices)
        model_kwargs = {'is_predictive' : False}
        
        kernel = NUTS(gplsm, target_accept_prob=adapt_delta)
        self.mcmc_ = MCMC(
            kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=1)

        rng_key = random.PRNGKey(self.random_state)
        self.mcmc_.run(rng_key, *model_args,  **model_kwargs)
        self.samples_ = self.mcmc_.get_samples()
        self.samples_ = jax.tree_map(lambda x : np.array(x), self.samples_)
        
        self.probas_ = self.predict_proba()
        self.auc_ = calculate_auc(self.Y_fit_, self.probas_)

        return self
    
    @property
    def model_args_(self):
        n_time_points, n_nodes, _ = self.samples_['U'].shape
        model_args = (None, self.time_points_, n_time_points, n_nodes, self.n_features,
                      self.length_scale, jnp.repeat(True, n_time_points))
        return model_args

    @property
    def model_kwargs_(self):
        return {'is_predictive': True}

    def predict_proba(self, time_points=None, random_state=0):
        rng_key = random.PRNGKey(random_state)
        n_samples, n_time_points, n_nodes, _ = self.samples_['X'].shape
        
        # re-calculate Xs from Ws
        if time_points is None:
            samples = {k: v for k, v in self.samples_.items() if k != 'X'}
        else:
            samples = self.samples_

        time_points = self.time_points_ if time_points is None else time_points
        model_args = (None, time_points, n_time_points, n_nodes, self.n_features,
                      self.length_scale, jnp.repeat(True, n_time_points))
        
        vmap_args = (samples, random.split(rng_key, n_samples))
        return vmap(
            lambda samples, rng_key : predict_proba(
                gplsm, rng_key, samples,
                *model_args, **self.model_kwargs_)
        )(*vmap_args).mean(axis=0)

    def predict_logits(self, time_points=None, random_state=0):
        rng_key = random.PRNGKey(random_state)
        n_samples, n_time_points, n_nodes, _ = self.samples_['X'].shape
        
        # re-calculate Xs from Ws
        if time_points is None:
            samples = {k: v for k, v in self.samples_.items() if k != 'X'}
        else:
            samples = self.samples_

        time_points = self.time_points_ if time_points is None else time_points
        model_args = (None, time_points, n_time_points, n_nodes, self.n_features,
                      self.length_scale, jnp.repeat(True, n_time_points))
        
        vmap_args = (self.samples_, random.split(rng_key, n_samples))
        return vmap(
            lambda samples, rng_key : predict_logit(
                gplsm, rng_key, samples,
                *model_args, **self.model_kwargs_)
        )(*vmap_args).mean(axis=0)
    
    def posterior_predictive(self, stat_fun, time_points=None, random_state=42):
        rng_key = random.PRNGKey(random_state)
        n_samples, n_time_points, n_nodes, _ = self.samples_['X'].shape
        
        # re-calculate Xs from Ws
        if time_points is None:
            samples = {k: v for k, v in self.samples_.items() if k != 'X'}
        else:
            samples = self.samples_

        time_points = self.time_points_ if time_points is None else time_points
        model_args = (None, time_points, n_time_points, n_nodes, self.n_features,
                      self.length_scale, jnp.repeat(True, n_time_points))
        
        vmap_args = (self.samples_, random.split(rng_key, n_samples))
        return np.asarray(vmap(
            lambda samples, rng_key : posterior_predictive(
                gplsm, rng_key, samples, stat_fun,
                *model_args, **self.model_kwargs_)
        )(*vmap_args))
