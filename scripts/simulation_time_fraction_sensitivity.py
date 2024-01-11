import plac
import os
import numpy as np
import pandas as pd


from scipy.stats import pearsonr
from scipy.special import logit
from sklearn.metrics import roc_auc_score

from splinetlsm import SplineDynamicLSM
from splinetlsm.datasets import synthetic_network_mixture
from splinetlsm.mcmc import dynamic_adjacency_to_vec
from splinetlsm.procrustes import longitudinal_procrustes_rotation


def simulation(seed, n_nodes=100, n_time_points=100, time_proportion=0.25, density=0.2): 
    seed = int(seed)
    n_nodes = int(n_nodes)
    n_time_points = int(n_time_points)
    time_proportion = float(time_proportion)
    density = float(density)
    
    Y, time_points, X, probas, U, coefs, intercept = synthetic_network_mixture(
        n_nodes=n_nodes, n_time_points=n_time_points,
        ls_type='gp', include_covariates=True, length_scale=0.2,
        tau=0.5, sigma=0.5, density=density, random_state=seed)
    y_true = dynamic_adjacency_to_vec(Y)

    model = SplineDynamicLSM(
        n_features=6, n_segments='auto', alpha=0.95, init_type='usvt', 
        max_time_points=Y.shape[0],
        random_state=4)
    model.fit(Y, time_points, X, 
        n_time_points=time_proportion, nonedge_proportion=2,
        step_size_power=0.75, step_size_delay=1, tol=1e-3, 
        max_iter=250)
    
    # parameter estimation
    subdiag = np.tril_indices(n_nodes, k=-1)
    UUt_rmse = 0.
    for t in range(n_time_points):
        UUt_true = (U[t] @ U[t].T)[subdiag]
        UUt_pred = (model.U_[t] @ model.U_[t].T)[subdiag]
        UUt_rmse += np.mean((UUt_true - UUt_pred) ** 2) / n_time_points
    UUt_rmse = np.sqrt(UUt_rmse)

    # compare smoothed latent positions with a single procrustes transform
    # only compare the top two dimensions ranked by the value of gamma_h
    #ids = np.argsort(model.gamma_)[::-1]
    U_pred, _ = longitudinal_procrustes_rotation(U, model.U_[..., :2])
    U_rmse = np.sqrt(np.mean((U - U_pred) ** 2))
    
    # coefficient estimation
    coefs_rmse = np.sum((model.coefs_ - coefs) ** 2, axis=1)
    coefs_rmse = np.sqrt(coefs_rmse.mean())
    intercept_rmse = np.sqrt(np.mean((model.intercept_ - intercept) ** 2))

    # total error for coefficientsa
    total_coefs_rmse = (np.sum((model.coefs_ - coefs) ** 2, axis=1) 
        + (model.intercept_ - intercept) ** 2)
    total_coefs_rmse = np.sqrt(total_coefs_rmse.mean())
    
    
    # log-odds estimation
    theta_rmse = np.sqrt(np.mean((logit(probas) - logit(model.probas_)) ** 2))

    data = {
        'density': y_true.mean(),
        'auc':  model.auc_,
        'ppc':  pearsonr(probas.ravel(), model.probas_.ravel())[0],
        'UUt_rmse': UUt_rmse,
        'U_rmse': U_rmse,
        'theta_rmse': theta_rmse,
        'coefs_rmse': coefs_rmse,
        'intercept_rmse': intercept_rmse,
        'total_coefs_rmse': total_coefs_rmse,
        'n_iter': model.n_iter_
    }
    data = pd.DataFrame(data, index=[0])

    out_file = f'result_{seed}.csv'
    dir_base = 'output_time_fraction_sensitivity'
    dir_name = os.path.join(dir_base, f"sim_n{n_nodes}_T{n_time_points}_d{density}_g{time_proportion}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(os.path.join(dir_name, out_file), index=False)


for n_time_points in [50, 100, 250, 500]:
    for gamma in [0.1, 0.25, 0.5, 0.8]:
        for i in range(50):
            simulation(seed=i, n_nodes=250, n_time_points=n_time_points, 
                    time_proportion=gamma)
            print('hi')
