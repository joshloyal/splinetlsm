import numpy as np

from scipy.special import expit
from sklearn.gaussian_process.kernels import RBF 
from sklearn.utils import check_random_state
from numpyro.distributions.util import vec_to_tril_matrix


__all__ = ['synthetic_network']


def tril_vec_to_matrix(x):
    A = vec_to_tril_matrix(x.astype(float), diagonal=-1)
    return A + A.T


def synthetic_network(n_nodes=50, n_time_points=20, n_features=2, intercept=-1, length_scale=0.1, random_state=42):
    time_points = np.arange(n_time_points) / (n_time_points - 1) 

    cov = RBF(length_scale=length_scale)(time_points.reshape(-1, 1))
    
    rng = check_random_state(random_state)

    X = rng.multivariate_normal(
            mean=np.zeros(n_time_points), cov=cov, size=(n_nodes, n_features))
    X = X.transpose((2, 0, 1))


    subdiag = np.tril_indices(n_nodes, k=-1)
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    Y = np.zeros((n_time_points, n_nodes, n_nodes))
    probas = np.zeros((n_time_points, n_dyads))
    for t in range(n_time_points):
        eta = intercept + (X[t] @ X[t].T)[subdiag]
        probas[t] = expit(eta)
        y_vec = rng.binomial(1, probas[t]) 
        Y[t] = tril_vec_to_matrix(y_vec)

    return Y, time_points, probas, X
