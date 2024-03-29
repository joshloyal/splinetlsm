import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
import scipy.sparse as sp

from jax import random, vmap
from numpyro.contrib.control_flow import scan
from math import ceil
from jax.scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state

from .bspline import bspline_basis
from .initialize import initialize_parameters, smooth_positions_procrustes
from ._svi import optimize_elbo_svi


def predict_proba_param(X, U, intercept, coefs):
    n_time_steps, n_nodes, _ = U.shape 
    subdiag = np.tril_indices(n_nodes, k=-1)

    probas = []
    for t in range(n_time_steps):
        eta = intercept[t] + (U[t] @ U[t].T)[subdiag]
        if X is not None:
            eta += (X[t] @ coefs[t])[subdiag]
        probas.append(expit(eta))
    
    return np.vstack(probas)


def predict_proba_sample(samples, X, B):
    n_nodes = samples['W'].shape[0]
    n_time_points = B.shape[-1]
    
    # calculate latent positions and coeficients
    U = samples['W'] @ B
    intercept = samples['W_intercept'] @ B

    if X is not None:
        coefs = samples['W_coefs'] @ B

    # calculate likelihood
    subdiag = jnp.tril_indices(n_nodes, k=-1)
    def probas_fun(carry, t):
        U = carry
        logits = intercept[t] + (U[..., t] @ U[..., t].T)[subdiag]
        if X is not None:
            logits += (X[t] @ coefs[..., t])[subdiag]
        return U, expit(logits)

    _, probas = scan(probas_fun, U, jnp.arange(n_time_points))

    return probas


def posterior_predictive(rng_key, samples, stat_fun, X, B):
    n_nodes = samples['W'].shape[0]
    n_time_points = B.shape[-1]
    
    # calculate latent positions and coeficients
    U = samples['W'] @ B
    intercept = samples['W_intercept'] @ B
    
    if X is not None:
        coefs = samples['W_coefs'] @ B

    # calculate likelihood
    subdiag = jnp.tril_indices(n_nodes, k=-1)
    def sample_network(carry, t):
        U = carry
        logits = intercept[t] + (U[..., t] @ U[..., t].T)[subdiag]
        if X is not None:
            logits += (X[t] @ coefs[..., t])[subdiag]
        y = dist.Bernoulli(logits=logits).sample(rng_key)
        return U, y

    _, y = scan(sample_network, U, jnp.arange(n_time_points))

    return stat_fun(y)


def calculate_auc(Y, probas):
    n_time_steps = len(Y)
    n_nodes = Y[0].shape[0]
    y_true, y_pred = [], []
    subdiag = np.tril_indices(n_nodes, k=-1)
    for t in range(n_time_steps):
        y_true.append(Y[t].toarray()[subdiag])
    
    return roc_auc_score(np.concatenate(y_true), probas.ravel())


class SplineDynamicLSM(object):
    """Spline Approximation to the Continuous-Time Dynamic Latent Space Model

    Parameters
    ----------
    n_features : int (default=2)
        The number of latent features. This is the dimension of the 
        latent space.

    n_segments : int (default='auto')
        The number of internal knots is the number of segments used to segment
        the unit interval minus one. By default the number of internal knots is
        min(ceil((n_nodes * n_time_steps) ** 0.2) + 1, 36).

    degree : int (default=3)
        The degree of the B-spline basis. The default is cubic B-splines.
    
    clamped : bool (default=False)
        Whether to use clamped B-splines. The default is non-clamped, that is,
        there are no repeated knots at the boundary.

    coefs_penalty_order : int (default=1)
        The order of the random-walk  prior for the coefficient function. All
        coordinate functions share the same order.

    ls_shape_prior : float (default=2)
        The shape parameter of the Gamma(shape/2, rate/2) prior on the nodewise
        variance parameters.
    
    ls_rate_prior : float (default=1)
        The rate parameter of the Gamma(shape/2, rate/2) prior on the nodewise
        variance parameters.
    
    coefs_shape_prior : float (default=2)
        The shape parameter of the Gamma(shape/2, rate/2) prior on the 
        variance parameters of the coefficient function.
    
    coefs_rate_prior : float (default=1)
        The rate parameter of the Gamma(shape/2, rate/2) prior on the 
        variance parameters of the coefficient function.
    
    mgp_a1 : float (default=2.)
        The shape parameter for the initial Gamma(a1, 1) prior for the
        multiplicative gamma process.
    
    mgp_a2 : float (default=3.)
        The shape parameter for the remaining Gamma(a2, 1) priors for the
        multiplicative gamma process.
    
    alpha : float (default=0.95)
        Fractional power of the alpha-variational posterior.
    
    init_type : str (default='usvt')
        Initialization method. The default is the USVT + ASE method, otherwise
        a random initialization is used.

    max_time_points : int (default=100)
        Maximum number of time points to subsample.

    random_state : int, RandomState instance or None (default=None)
        Controls the random seed given to the method. Pass an int for 
        reproducible output across multiple function calls.

    
    Examples
    --------

    >>> from splinetlsm import SplineDynamicLSM  
    >>> from dynetlsm.datasets import load_polecat
    >>> Y, time_points, X, _, _, _, _ = load_polecat()
    >>> model = SplineDynamicLSM(n_features=6).fit(Y, time_points, X)
    """
    def __init__(self,
                 n_features=2,
                 n_segments='auto',
                 degree=3,
                 clamped=False,
                 coefs_penalty_order=1,
                 ls_shape_prior=2.,
                 ls_rate_prior=1.,
                 coefs_shape_prior=2.,
                 coefs_rate_prior=1.,
                 mgp_a1=2.,
                 mgp_a2=3.,
                 alpha=0.95,
                 init_type='usvt',
                 max_time_points=100,
                 random_state=42):
        self.n_features = n_features
        self.n_segments = n_segments
        self.degree = degree
        self.clamped = clamped
        self.coefs_penalty_order = coefs_penalty_order
        self.ls_rate_prior = ls_rate_prior
        self.ls_shape_prior = ls_shape_prior
        self.coefs_rate_prior = coefs_rate_prior
        self.coefs_shape_prior = coefs_shape_prior
        self.mgp_a1 = mgp_a1
        self.mgp_a2 = mgp_a2
        self.alpha = alpha
        self.init_type = init_type
        self.max_time_points = max_time_points
        self.random_state = random_state

    def fit(self, Y, time_points, X=None, 
            nonedge_proportion=2, n_time_points=0.25, n_samples=2000,
            step_size_delay=1,  step_size_power=0.75, 
            max_iter=250, tol=1e-3):
        """Infer the approximate alpha-variational posterior using SVI
        based on dynamic network Y and covariates X.

        Parameters
        ----------
        Y : list of length n_time_points of (n_nodes, n_nodes) sparse csc matrices
           A list of adjacency matrices stored as sparse csc matrices. 

        time_points : (n_time_points,) ndarray of observed time points
           The observed time points. 
        
        X : (n_time_points, n_nodes, n_nodes, n_covariates) ndarray of covariates
            A tensor of dyadic covariates. Note that an intercept is included 
            automatically, so a feature of ones does not need to be included.
            Set to `None` if no dyadic covariates are present, this will 
            still include an intercept.
        
        nonedge_proportion : int (default=2)
            The non-edge fraction used to construct the subsample used to
            estimate the stochastic gradients.
        
        n_time_points : float (default=0.25)
            The temporal snapshot fraction.

            - If an int, then uses n_time_point time points.
            - If a float, then uses that fraction of total time points.

        step_size_delay : float (default=1)
            learning_rate = (step_size_delay + i)^(-step_size_power).

        step_size_power : float (default=0.75)
            learning_rate = (step_size_delay + i)^(-step_size_power).

        max_iter : int (default=25)
            Maximum number of iterations to run the SVI algorithm.

        tol : float (default=1e-3)
            Tolerance to determine convergence of the SVI algorithm.
        """
        if isinstance(Y, np.ndarray):
            self.Y_fit_ = []
            for t in range(Y.shape[0]):
                self.Y_fit_.append(sp.csc_matrix(Y[t]))
        else:
            self.Y_fit_ = Y
        
        # scale time_points to [0, 1] interval
        self.time_min_ = np.min(time_points)
        self.time_max_ = np.max(time_points)
        self.time_points_ = ((time_points - self.time_min_) / 
            (self.time_max_ - self.time_min_))

        n_time_steps = self.time_points_.shape[0]
        n_nodes = Y[0].shape[0]
        
        if np.asarray(n_time_points).dtype.kind == 'f':
            self.n_time_points_ = max(5,
                min(ceil(n_time_steps * n_time_points), self.max_time_points))
        else:
            self.n_time_points_ = min(n_time_points, n_time_steps)
        
        if self.n_segments == 'auto':
            self.n_segments_ = max(5, min(
                    ceil((n_nodes * n_time_steps) ** 0.2) + 1, 36))
        else:
            self.n_segments_ = self.n_segments
        
        self.X_fit_ = X
        self.B_fit_ = bspline_basis(
                self.time_points_, n_segments=self.n_segments_, 
                degree=self.degree, clamped=self.clamped)
        
        if self.init_type == 'usvt':
            W_init, W_coefs_init = initialize_parameters(
                self.Y_fit_, self.B_fit_, X=self.X_fit_, 
                n_features=self.n_features, random_state=self.random_state)
        else: 
            rng = check_random_state(self.random_state)     
            W0 = rng.randn(n_nodes, self.n_features, 1)
            W_init = W0 + np.cumsum(
                    0.01 * rng.randn(n_nodes, self.n_features, self.B_fit_.shape[0]),
                    axis=-1)
            W_init = W_init.transpose((0, 2, 1))

            n_covariates = 1 if X is None else 1 + self.X_fit_.shape[-1]
            W0 = rng.randn(n_covariates, 1)
            W_coefs_init = W0 + np.cumsum(
                    0.01 * rng.randn(n_covariates, self.B_fit_.shape[0]),
                    axis=-1)
            W_coefs_init = W_coefs_init.T
    
        params, moments, diagnostics = optimize_elbo_svi(
                self.Y_fit_, self.B_fit_, self.time_points_, self.X_fit_,
                alpha=self.alpha,
                W_init=W_init, W_coefs_init=W_coefs_init,
                n_features=self.n_features,
                penalty_order=1,
                coefs_penalty_order=self.coefs_penalty_order,
                rate_prior=self.ls_rate_prior, shape_prior=self.ls_shape_prior,
                coefs_rate_prior=self.coefs_rate_prior, 
                coefs_shape_prior=self.coefs_shape_prior,
                mgp_a1=self.mgp_a1, mgp_a2=self.mgp_a2, 
                tau_prec=1., coefs_tau_prec=1e-2,
                nonedge_proportion=nonedge_proportion, 
                n_time_steps=self.n_time_points_, 
                step_size_delay=step_size_delay,
                step_size_power=step_size_power, max_iter=max_iter,
                tol=tol, random_state=self.random_state)
        
        # unpack convergence diagnostics
        self.converged_ = diagnostics['converged']
        self.n_iter_ = diagnostics['n_iter']
        self.diffs_ = diagnostics['diffs']
        self.loglik_ = diagnostics['loglik']
        self.step_size_ = diagnostics['step_size']

        # unpack parameters and moments
        self.W_ = params['W']
        self.W_sigma_ = params['W_sigma']
        self.b_ = params['b']
        self.w_prec_ = moments['w_prec']

        # extract the latent positions and smooth them over-time through an
        # procrustes re-alignment.
        self.U_ = smooth_positions_procrustes(
                moments['U'].transpose((1, 0, 2)))

        self.W_intercept_ = params['W_intercept']
        self.W_intercept_sigma_ = params['W_intercept_sigma']
        self.b_intercept_ = params['b_intercept']
        self.w_intercept_prec_ = moments['w_intercept_prec']
        self.intercept_ = moments['intercept']
        
        if self.X_fit_ is not None:
            self.W_coefs_ = params['W_coefs']
            self.W_coefs_sigma_ = params['W_coefs_sigma']
            self.b_coefs_ = params['b_coefs']
            self.w_coefs_prec_ = moments['w_coefs_prec']
            self.coefs_ = moments['coefs'].T
        else:
            self.coefs_ = None
        
        self.gamma_ = moments['gamma']

        self.mgp_rate_ = params['mgp_rate']
        self.mgp_shape_ = params['mgp_shape']

        self.a_ = params['a']
        self.p_ = params['p']

        self.a_coefs_ = params['a_coefs']
        self.p_coefs_ = params['p_coefs']
        
        # sample spline coefficients from the variational posterior
        self.samples_ = self.sample(n_samples=n_samples)
        
        # calculate in-sample AUC
        self.probas_ = self.predict_proba()
        self.auc_ = calculate_auc(self.Y_fit_, self.probas_)
        
        return self
    
    def sample(self, n_samples=2000, random_state=0):
        samples = dict()

        rng_key1, rng_key2 = random.split(random.PRNGKey(random_state), 2)
        
        W = self.W_.transpose((0,2,1))
        W_sigma = self.W_sigma_.transpose((3,0,1,2))
        samples['W'] = dist.MultivariateNormal(
                loc=W, covariance_matrix=W_sigma).sample(
                        rng_key1, sample_shape=(n_samples,))
        
        samples['W_intercept'] = dist.MultivariateNormal(
                loc=self.W_intercept_, 
                covariance_matrix=self.W_intercept_sigma_).sample(
                rng_key2, sample_shape=(n_samples,))

        if self.X_fit_ is not None:
            W_coefs = self.W_coefs_.T
            W_coefs_sigma = self.W_coefs_sigma_.transpose((2,0,1))
            samples['W_coefs'] = dist.MultivariateNormal(
                    loc=W_coefs, covariance_matrix=W_coefs_sigma).sample(
                            rng_key2, sample_shape=(n_samples,))

        return samples
    
    def predict_proba(self, estimator='param_mean'):
    
        X = None if self.X_fit_ is None else jnp.array(self.X_fit_)
        
        if estimator == 'param_mean':
            return predict_proba_param(X, self.U_, self.intercept_, self.coefs_)
        
        return vmap(
            lambda samples : predict_proba_sample(samples, 
                X, jnp.array(self.B_fit_.todense()))
            )(self.samples_).mean(axis=0)
    
    def posterior_predictive(self, stat_fun, chunk_size=None, random_state=42):
        if chunk_size is not None:
            return self._chunked_posterior_predictive(
                    stat_fun, chunk_size=chunk_size, random_state=random_state)

        rng_key = random.PRNGKey(random_state)
        
        X = None if self.X_fit_ is None else jnp.array(self.X_fit_)
        n_samples  = self.samples_['W'].shape[0]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))
        return np.asarray(vmap(
            lambda samples, rng_key : posterior_predictive(
                rng_key, samples, stat_fun,
                X, jnp.array(self.B_fit_.todense()))
            )(*vmap_args))

    def _chunked_posterior_predictive(self, stat_fun, chunk_size=100, random_state=42):
        rng_key = random.PRNGKey(random_state)
        
        X = None if self.X_fit_ is None else jnp.array(self.X_fit_)
        n_samples  = self.samples_['W'].shape[0]
        
        out = []
        n_chunks = ceil(n_samples / chunk_size)
        for idx in range(n_chunks):
            start = idx * chunk_size
            end = start + chunk_size
            chunked_samples = {
                    k: v[start:end] for (k, v) in self.samples_.items()
            }

            vmap_args = (chunked_samples, random.split(rng_key, chunk_size))
            out.append(np.asarray(vmap(
                lambda samples, rng_key : posterior_predictive(
                    rng_key, samples, stat_fun,
                    X, jnp.array(self.B_fit_.todense()))
                )(*vmap_args)))

        return np.vstack(out)

