import numpy as np

from scipy.special import expit
from scipy.optimize import root_scalar
from sklearn.gaussian_process.kernels import RBF 
from sklearn.utils import check_random_state
from numpyro.distributions.util import vec_to_tril_matrix

from ..bspline import bspline_basis
from ..static_gof import vec_to_adjacency

__all__ = ['synthetic_network', 'synthetic_network_mixture']


def tril_vec_to_matrix(x):
    A = vec_to_tril_matrix(x.astype(float), diagonal=-1)
    return A + A.T


def generate_gp(time_points, n_nodes=100, n_features=2, length_scale=0.2, tau=0.25, random_state=None):
    rng = check_random_state(random_state)
    
    # RBF GP
    n_time_points = time_points.shape[0]
    cov = RBF(length_scale=length_scale)(time_points.reshape(-1, 1)) 
    U = tau * rng.multivariate_normal(
            mean=np.zeros(n_time_points), cov=cov, size=(n_nodes, n_features))
    U = U.transpose((2, 0, 1))
    
    return U


def generate_gp_coefs(time_points, n_covariates=2, length_scale=0.2, tau=0.1, random_state=None):
    rng = check_random_state(random_state)
    
    # RBF GP
    n_time_points = time_points.shape[0]
    cov = RBF(length_scale=length_scale)(time_points.reshape(-1, 1)) 
    U = tau * rng.multivariate_normal(
            mean=np.zeros(n_time_points), cov=cov, size=(n_covariates,))
    
    return U.T


def generate_bspline(time_points, 
        n_nodes=100, n_features=2, n_segments=11, degree=3, 
        tau=4, sigma=0.1, random_state=None):
    rng = check_random_state(random_state)
    
    B = bspline_basis(
        time_points, n_segments=n_segments, degree=degree, return_sparse=False)
     
    # Gaussian Random-Walk
    W0 = rng.randn(n_nodes, n_features, 1)
    W = W0 + np.cumsum(
            sigma * rng.randn(n_nodes, n_features, B.shape[0]), 
            axis=-1)

    return tau * (W @ B).transpose((2, 0, 1))



def synthetic_network(n_nodes=50, n_time_points=20, n_features=2, intercept=-4, 
        ls_type='bspline', include_covariates=False, length_scale=0.2, 
        tau=2, sigma=0.05, random_state=42, density=0.25):
    rng = check_random_state(random_state)
    time_points = np.arange(n_time_points) / (n_time_points - 1) 

    #cov = 2 * RBF(length_scale=length_scale)(time_points.reshape(-1, 1))
    #
    #rng = check_random_state(random_state)
    #
    #U = rng.multivariate_normal(
    #        mean=np.zeros(n_time_points), cov=cov, size=(n_nodes, n_features))
    #U = U.transpose((2, 0, 1))
    
    if ls_type == 'bspline':
        U = generate_bspline(
            time_points, n_nodes=n_nodes, n_features=2, 
            tau=tau, sigma=sigma, random_state=rng)
    else:
        U = generate_gp(
            time_points, n_nodes=n_nodes, n_features=2, 
            length_scale=length_scale, tau=tau, 
            random_state=rng)
    
    # covariates
    if include_covariates:
        X = np.zeros((n_time_points, n_nodes, n_nodes, 2))
        for p in range(2):
            x = rng.randn(n_dyads)
            for t in range(n_time_points):
                X[t, ..., p] = vec_to_adjacency(x)
        coefs = np.array([0.5, -0.5])
    else:
        X = None


    subdiag = np.tril_indices(n_nodes, k=-1)
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    Y = np.zeros((n_time_points, n_nodes, n_nodes))
    probas = np.zeros((n_time_points, n_dyads))
    for t in range(n_time_points):
        eta = intercept + (U[t] @ U[t].T)[subdiag]
        if include_covariates:
            eta += (X[t] @ coefs)[subdiag]
        probas[t] = expit(eta)
        y_vec = rng.binomial(1, probas[t]) 
        Y[t] = tril_vec_to_matrix(y_vec)

    return Y, time_points, X, probas, U


def find_intercept(logits, target_density):
    def density_func(intercept):
        return expit(logits + intercept).mean() - target_density

    return root_scalar(density_func, bracket=[-10, 10]).root


def synthetic_network_mixture(n_nodes=50, n_time_points=20, density=0.25, 
        include_covariates=False, ls_type='bspline',
        tau=0.25, sigma=0.25, length_scale=0.2, random_state=42):
    
    rng = check_random_state(random_state)
    time_points = np.arange(n_time_points) / (n_time_points - 1) 
    
    if ls_type == 'bspline':
        U = generate_bspline(
            time_points, n_nodes=n_nodes, n_features=2, 
            tau=tau, sigma=sigma, random_state=rng)
    else:
        U = generate_gp(
            time_points, n_nodes=n_nodes, n_features=2, 
            length_scale=length_scale, tau=tau, random_state=rng)
 
    # latent space
    centers = np.array([[1.5, 0],
                        [-1.5, 0],
                        [0., 1.]])
    z = rng.choice([0, 1, 2], size=n_nodes)
    for t in range(n_time_points):
        U[t] += centers[z]
    
    # covariates
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    if include_covariates: 
        X = np.zeros((n_time_points, n_nodes, n_nodes, 2))
        for p in range(2):
            x = rng.randn(n_dyads)
            for t in range(n_time_points):
                X[t, ..., p] = vec_to_adjacency(x)
        coefs = np.array([1., -1.]) + generate_gp_coefs(
                time_points, n_covariates=2, 
                length_scale=length_scale, tau=tau,
                random_state=rng) 
    else:
        X = None
        coefs = None


    subdiag = np.tril_indices(n_nodes, k=-1)
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    Y = np.zeros((n_time_points, n_nodes, n_nodes))
    probas = np.zeros((n_time_points, n_dyads))
    intercept = np.zeros(n_time_points)
    for t in range(n_time_points):
        eta = (U[t] @ U[t].T)[subdiag]
        if include_covariates:
            eta += (X[t] @ coefs[t])[subdiag]
        intercept[t] = find_intercept(eta, target_density=density)

        probas[t] = expit(eta + intercept[t])
        y_vec = rng.binomial(1, probas[t]) 
        Y[t] = tril_vec_to_matrix(y_vec)

    return Y, time_points, X, probas, U, coefs, intercept
