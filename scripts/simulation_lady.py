import plac
import os
import numpy as np
import pandas as pd
import time

from scipy.stats import pearsonr
from scipy.special import logit
from sklearn.metrics import roc_auc_score

from splinetlsm import SplineDynamicLSM
from splinetlsm.lady import LadyNetworkModel 
from splinetlsm.datasets import synthetic_network_mixture
from splinetlsm.mcmc import dynamic_adjacency_to_vec
from splinetlsm.procrustes import longitudinal_procrustes_rotation


def simulation(seed, n_nodes=100, n_time_points=10, ls_type='gp', density=0.2, n_features=2): 
    seed = int(seed)
    n_nodes = int(n_nodes)
    n_time_points = int(n_time_points)
    density = float(density)
    n_features = int(n_features)
    
    Y, time_points, X, probas, U, coefs, intercept = synthetic_network_mixture(
        n_nodes=n_nodes, n_time_points=n_time_points,
        ls_type=ls_type, include_covariates=False, length_scale=0.2,
        tau=0.5, sigma=0.5, density=density, random_state=seed)
    y_true = dynamic_adjacency_to_vec(Y)
    
    t = time.time()
    model_lady = LadyNetworkModel(n_features=n_features)
    model_lady.sample(Y, n_warmup=2500, n_samples=2500)
    lady_time = time.time() - t

    data = {
        'density': y_true.mean(),
        'auc_lady':  model_lady.auc_,
        'ppc_lady':  pearsonr(probas.ravel(), model_lady.probas_.ravel())[0],
        'time_lady': lady_time,
    }
    data = pd.DataFrame(data, index=[0])

    out_file = f'result_{seed}.csv'
    dir_base = 'output_comparison'
    dir_name = os.path.join(dir_base, f"lady_n{n_nodes}_T{n_time_points}_d{density}")

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(os.path.join(dir_name, out_file), index=False)


# NOTE: This is meant to be run in parallel on a computer cluster!
n_reps = 50

for density in [0.1, 0.2, 0.3]:
    for i in range(50):
        simulation(seed=i, n_nodes=100, n_time_points=10, density=density)

    for i in range(50):
        simulation(seed=i, n_nodes=100, n_time_points=20, density=density)
    
    for i in range(50):
        simulation(seed=i, n_nodes=200, n_time_points=10, density=density)

    plac.call(simulation)
