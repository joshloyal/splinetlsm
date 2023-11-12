import plac
import os
import numpy as np
import pandas as pd
import time

from scipy.stats import pearsonr
from scipy.special import logit
from sklearn.metrics import roc_auc_score

from splinetlsm import SplineDynamicLSM
from splinetlsm.gp import GaussianProcessDynamicLSM
from splinetlsm.datasets import synthetic_network_mixture
from splinetlsm.mcmc import dynamic_adjacency_to_vec
from splinetlsm.procrustes import longitudinal_procrustes_rotation


def simulation(seed, n_nodes=200, n_time_points=10, ls_type='gp', density=0.2):
    seed = int(seed)
    n_nodes = int(n_nodes)
    n_time_points = int(n_time_points)
    density = float(density)

    Y, time_points, X, probas, U, coefs, intercept = synthetic_network_mixture(
        n_nodes=n_nodes, n_time_points=n_time_points,
        ls_type=ls_type, include_covariates=False, length_scale=0.2,
        tau=0.5, sigma=0.5, density=density, random_state=seed)
    
    out_dir = f'./data_{seed}'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    np.savetxt(os.path.join(out_dir, 'time_points.npy'), time_points)
    for t in range(Y.shape[0]):
        np.savetxt(os.path.join(out_dir, f'Y_{t+1}.npy'), Y[t])
        np.savetxt(os.path.join(out_dir, f'proba_{t+1}.npy'), probas[t])

if __name__ == '__main__':
    simulation(10)
