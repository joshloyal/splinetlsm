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


def simulation(seed, n_nodes=100, n_time_points=10, density=0.2): 
    seed = int(seed)
    n_nodes = int(n_nodes)
    n_time_points = int(n_time_points)
    density = float(density)
    
    Y, time_points, X, probas, U, coefs, intercept = synthetic_network_mixture(
        n_nodes=n_nodes, n_time_points=n_time_points,
        ls_type='gp', include_covariates=False, length_scale=0.2,
        tau=0.5, sigma=0.5, density=density, random_state=seed)
    y_true = dynamic_adjacency_to_vec(Y)
    
    t = time.time()
    model_gp = GaussianProcessDynamicLSM(n_features=6)
    model_gp.sample(Y, time_points, n_warmup=2500, n_samples=2500)
    gp_time = time.time() - t

    t = time.time()
    model = SplineDynamicLSM(
        n_features=6, n_segments='auto', alpha=0.95, init_type='usvt', 
        random_state=4)
    model.fit(Y, time_points, X, 
        n_time_points=0.25, nonedge_proportion=2,
        step_size_power=0.75, step_size_delay=1, tol=1e-3, 
        max_iter=250)
    svi_time = time.time() - t
    
    data = {
        'density': y_true.mean(),
        'auc_gp':  model_gp.auc_,
        'ppc_gp':  pearsonr(probas.ravel(), model_gp.probas_.ravel())[0],
        'logit_gp': np.sqrt(np.mean((logit(probas) - model_gp.predict_logits()) ** 2)),
        'time_gp': gp_time,
        'auc_svi':  model.auc_,
        'ppc_svi':  pearsonr(probas.ravel(), model.probas_.ravel())[0],
        'logit_svi': np.sqrt(np.mean((logit(probas) - logit(model_gp.probas_)) ** 2)),
        'time_svi': svi_time
    }
    data = pd.DataFrame(data, index=[0])

    out_file = f'result_{seed}.csv'
    dir_base = 'output_comparison'
    dir_name = os.path.join(dir_base, f"sim_n{n_nodes}_T{n_time_points}_d{density}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(os.path.join(dir_name, out_file), index=False)


# NOTE: This is meant to be run in parallel on a computer cluster!
n_reps = 50

for density in [0.1, 0.2, 0.3]:
    for i in range(50):
        simulation(seed=i, n_nodes=100, n_time_points=10, density=density)
        print('hi')

    for i in range(50):
        simulation(seed=i, n_nodes=100, n_time_points=20, density=density)
    
    for i in range(50):
        simulation(seed=i, n_nodes=200, n_time_points=10, density=density)

