import os
import numpy as np
import pandas as pd


from scipy.stats import pearsonr
from scipy.special import logit
from scipy.spatial import procrustes
from sklearn.metrics import roc_auc_score

from splinetlsm import SplineDynamicLSM
from splinetlsm.datasets import synthetic_network_mixture
from splinetlsm.mcmc import dynamic_adjacency_to_vec
from splinetlsm.procrustes import longitudinal_procrustes_rotation


def simulation(seed, n_nodes=100, n_time_points=100, ls_type='gp',  nu=0.5, density=0.2): 
    seed = int(seed)
    n_nodes = int(n_nodes)
    n_time_points = int(n_time_points)
    density = float(density)
    nu = float(nu)
    
    Y, time_points, X, probas, U, coefs, intercept, z = synthetic_network_mixture(
        n_nodes=n_nodes, n_time_points=n_time_points, n_features=2,
        ls_type=ls_type, include_covariates=True, length_scale=0.2, nu=nu,
        tau=0.5, sigma=0.5, density=density, random_state=seed)
    y_true = dynamic_adjacency_to_vec(Y)

    model = SplineDynamicLSM(
        n_features=6, n_segments='auto', alpha=0.95, init_type='usvt', 
        random_state=4)
    model.fit(Y, time_points, X, 
        n_time_points=0.25, nonedge_proportion=2,
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
    U_pred, _ = longitudinal_procrustes_rotation(U, model.U_[..., :2])
    U_rmse = np.sqrt(np.mean((U - U_pred) ** 2))
    
    # pad true U matrix with zeros
    U_padded = np.concatenate([U, np.zeros((U.shape[0], U.shape[1], 4))], axis=2)
    U_pred_all, _ = longitudinal_procrustes_rotation(U_padded, model.U_)
    U_rmse_all = np.sqrt(np.mean((U_padded - U_pred_all)** 2))
     
    # dimension selection 
    d_max = max(model.n_features_, 2)
    U_padded_select = U_padded[..., :d_max]
    U_pred, _ = longitudinal_procrustes_rotation(
        U_padded_select, model.U_[..., :d_max])
    U_rmse_select = np.sqrt(np.mean((U_padded_select - U_pred) ** 2))
    
    # procrustes correlation
    proc_corr_all = 0.
    proc_corr = 0.
    proc_corr_select = 0.
    for t in range(n_time_points):
        proc_corr += np.sqrt(1 - procrustes(U[t], model.U_[t, :, :2])[-1]) / n_time_points
        proc_corr_all += np.sqrt(1 - procrustes(U_padded[t], model.U_[t])[-1]) / n_time_points
        proc_corr_select += np.sqrt(1 - procrustes(U_padded_select[t], model.U_[t, :, :d_max])[-1]) / n_time_points

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
        'U_rmse_all': U_rmse_all,
        'U_rmse_select': U_rmse_select,
        'proc_corr': proc_corr,
        'proc_corr_all': proc_corr_all,
        'proc_corr_select': proc_corr_select,
        'theta_rmse': theta_rmse,
        'coefs_rmse': coefs_rmse,
        'intercept_rmse': intercept_rmse,
        'total_coefs_rmse': total_coefs_rmse,
        'n_iter': model.n_iter_,
        'n_features': model.n_features_
    }
    data = pd.DataFrame(data, index=[0])

    out_file = f'result_{seed}.csv'
    dir_base = 'output_smoothness'
    dir_name = os.path.join(dir_base, 'output', f"{ls_type}_nu{nu}_n{n_nodes}_T{n_time_points}_d{density}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(os.path.join(dir_name, out_file), index=False)


n_reps = 50
for gp, nu in [('gp', 0.5), ('matern', 2.5), ('matern', 1.5), ('matern', 0.5)]:
    for i in range(n_reps):
        simulation(seed=i, n_nodes=200, n_time_points=100, ls_type=gp, nu=nu)
        print(i)
