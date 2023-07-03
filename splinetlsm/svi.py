import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist

from jax import random, vmap
from numpyro.contrib.control_flow import scan
from math import ceil
from jax.scipy.special import expit
from sklearn.metrics import roc_auc_score

from .bspline import bspline_basis
from ._svi import optimize_elbo_svi


#def predict_probas(X, moments):
#    n_nodes, n_time_steps, _ = moments['U'].shape 
#    subdiag = np.tril_indices(n_nodes, k=-1)
#
#    probas = []
#    for t in range(n_time_steps):
#        U = moments['U'][:, t, :]
#        coefs = moments['coefs'][:, t]
#
#        probas.append(expit((X[t] @ coefs + U @ U.T)[subdiag]))
#    
#    return np.vstack(probas)


def predict_proba(samples, X, B):
    n_nodes = samples['W'].shape[0]
    n_time_points = B.shape[-1]
    
    # calculate latent positions and coeficients
    U = samples['W'] @ B
    coefs = samples['W_coefs'] @ B

    # calculate likelihood
    subdiag = jnp.tril_indices(n_nodes, k=-1)
    def probas_fun(carry, t):
        U = carry
        logits = (X[t] @ coefs[..., t] + U[..., t] @ U[..., t].T)[subdiag]
        return U, expit(logits)

    _, probas = scan(probas_fun, U, jnp.arange(n_time_points))

    return probas


def posterior_predictive(rng_key, samples, stat_fun, X, B):
    n_nodes = samples['W'].shape[0]
    n_time_points = B.shape[-1]
    
    # calculate latent positions and coeficients
    U = samples['W'] @ B
    coefs = samples['W_coefs'] @ B

    # calculate likelihood
    subdiag = jnp.tril_indices(n_nodes, k=-1)
    def sample_network(carry, t):
        U = carry
        logits = (X[t] @ coefs[..., t] + U[..., t] @ U[..., t].T)[subdiag]
        y = dist.Bernoulli(logits=logits).sample(rng_key)
        return U, y

    _, y = scan(sample_network, U, jnp.arange(n_time_points))

    return stat_fun(y)


def calculate_auc(Y, X, moments):
    n_time_steps = len(Y)
    n_nodes = Y[0].shape[0]
    y_true, y_pred = [], []
    subdiag = np.tril_indices(n_nodes, k=-1)
    for t in range(n_time_steps):
        U = moments['U'][:, t, :]
        coefs = moments['coefs'][:, t]

        y_true.append(Y[t].toarray()[subdiag])
        y_pred.append(expit((X[t] @ coefs + U @ U.T)[subdiag]))
    
    return roc_auc_score(np.concatenate(y_true), np.concatenate(y_pred))


class SplineDynamicLSM(object):
    def __init__(self,
                 n_features=2,
                 n_knots='auto',
                 degree=3,
                 penalty_order=1,
                 coefs_penalty_order=1,
                 rate_prior=2.,
                 shape_prior=1.,
                 coefs_rate_prior=2.,
                 coefs_shape_prior=1.,
                 mgp_a1=2.,
                 mgp_a2=3.,
                 tau_prec=1e-2,
                 coefs_tau_prec=1e-2,
                 random_state=42):
        self.n_features = n_features
        self.n_knots = n_knots
        self.degree = degree
        self.penalty_order = penalty_order
        self.coefs_penalty_order = coefs_penalty_order
        self.rate_prior = rate_prior
        self.shape_prior = shape_prior
        self.coefs_rate_prior = coefs_rate_prior
        self.coefs_shape_prior = coefs_shape_prior
        self.mgp_a1 = mgp_a1
        self.mgp_a2 = mgp_a2
        self.tau_prec = tau_prec
        self.coefs_tau_prec = coefs_tau_prec
        self.random_state = random_state

    def fit(self, Y, time_points, X=None, 
            nonedge_proportion=1, n_time_points=0.5, n_samples=2000,
            step_size_delay=1,  step_size_power=0.75, max_iter=1000, tol=1e-3):
        """
        Parameters
        ----------
        Y : list of length T of (n,n) sparse csc matrices
        
        time_points : (T,) ndarray of observed time points
        
        X : (T,n,n,p) covariate matrix.

        step_size_delay : float
            (step_size_delay + i)^(-step_size_power)

        step_size_power : float
            (step_size_delay + i)^(-step_size_power)

        max_iter : int
            Maximum number of iterations to run SVI

        tol : float
            Tolerance to determine convergence of the SVI algorithm
        """

        n_time_steps = time_points.shape[0]
        n_nodes = Y[0].shape[0]
        
        if np.asarray(n_time_points).dtype.kind == 'f':
            self.n_time_points_ = max(ceil(n_time_steps * n_time_points), 100)
        else:
            self.n_time_points_ = min(n_time_points, n_time_steps)
        
        if self.n_knots == 'auto':
            self.n_knots_ = (n_time_steps if n_time_steps < 25 else 
                ceil(min(n_time_steps / 4, 40)))
        else:
            self.n_knots_ = self.n_knots
        
        if X is None:
            # intercept
            self.X_fit_ = np.ones((n_time_steps, n_nodes, n_nodes, 1))
        else:
            self.X_fit_ = X

        self.time_points_ = time_points
        self.B_fit_, self.bs_ = bspline_basis(
                self.time_points_, n_knots=self.n_knots_, degree=self.degree)
        
        self.params_, self.moments_, diagnostics = optimize_elbo_svi(
                Y, self.B_fit_, time_points, self.X_fit_,
                n_features=self.n_features,
                penalty_order=self.penalty_order,
                coefs_penalty_order=self.coefs_penalty_order,
                rate_prior=self.rate_prior, shape_prior=self.shape_prior,
                coefs_rate_prior=self.coefs_rate_prior, 
                coefs_shape_prior=self.coefs_shape_prior,
                mgp_a1=self.mgp_a1, mgp_a2=self.mgp_a2, 
                tau_prec=self.tau_prec, coefs_tau_prec=self.coefs_tau_prec,
                nonedge_proportion=nonedge_proportion, 
                n_time_steps=self.n_time_points_, 
                step_size_delay=step_size_delay,
                step_size_power=step_size_power, max_iter=max_iter,
                tol=tol, random_state=self.random_state)
     
        self.converged_ = diagnostics['converged']
        self.n_iter_ = diagnostics['n_iter']
        self.diffs_ = diagnostics['diffs']
        
        # sample spline coefficients from the variational posterior
        self.samples_ = self.sample(n_samples=n_samples)
        
        # calculate in-sample AUC
        #self.probas_ = predict_probas(self.X_fit_, self.moments_)
        self.probas_ = self.predict_proba()
        self.auc_ = calculate_auc(Y, self.X_fit_, self.moments_)
        

        return self
    
    def sample(self, n_samples=2000, random_state=0):
        samples = dict()

        rng_key1, rng_key2 = random.split(random.PRNGKey(random_state), 2)
        
        W = self.params_['W'].transpose((0,2,1))
        W_sigma = self.params_['W_sigma'].transpose((3,0,1,2))

        samples['W'] = dist.MultivariateNormal(
                loc=W, covariance_matrix=W_sigma).sample(
                        rng_key1, sample_shape=(n_samples,))
        
        W_coefs = self.params_['W_coefs'].T
        W_coefs_sigma = self.params_['W_coefs_sigma'].transpose((2,0,1))

        samples['W_coefs'] = dist.MultivariateNormal(
                loc=W_coefs, covariance_matrix=W_coefs_sigma).sample(
                        rng_key2, sample_shape=(n_samples,))

        return samples
    
    def predict_proba(self):
        return vmap(
            lambda samples : predict_proba(samples, 
                jnp.array(self.X_fit_), jnp.array(self.B_fit_.todense()))
            )(self.samples_).mean(axis=0)
    
    def posterior_predictive(self, stat_fun, random_state=42):
        rng_key = random.PRNGKey(random_state)
        
        n_samples  = self.samples_['W'].shape[0]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))
        return np.asarray(vmap(
            lambda samples, rng_key : posterior_predictive(
                rng_key, samples, stat_fun,
                jnp.array(self.X_fit_), jnp.array(self.B_fit_.todense()))
            )(*vmap_args))
