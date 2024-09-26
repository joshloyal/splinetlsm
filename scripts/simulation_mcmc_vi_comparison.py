import numpy as np
import functools
import pandas as pd
import os

from splinetlsm.dynamic_gof import density, transitivity, node_degree
from splinetlsm.svi import SplineDynamicLSM
from splinetlsm.hmc import SplineDynamicLSMHMC
from splinetlsm.mcmc import dynamic_adjacency_to_vec
from splinetlsm.datasets import synthetic_network_mixture


def simulation(seed):
    seed = int(seed)
    n_nodes = 100
    n_time_points = 10

    Y, time_points, X, probas, Z, coefs_true, intercept = synthetic_network_mixture(
        n_nodes=n_nodes, n_time_points=n_time_points,
        ls_type='gp',
        include_covariates=True, tau=0.5, sigma=0.5,
        density=0.2, length_scale=0.2, random_state=seed)
    
    # HMC
    model = SplineDynamicLSMHMC(n_features=6, alpha=0.95, random_state=42)
    model.sample(Y, time_points, X, n_warmup=2500, n_samples=2500)
    
    # SVI
    svi = SplineDynamicLSM(n_features=6, n_segments='auto',
                         alpha=0.95, init_type='usvt',
                         random_state=4)
    svi.fit(Y, time_points, X=X,
              n_time_points=0.25, nonedge_proportion=2,
              step_size_power=0.75, step_size_delay=1,
              tol=1e-3, max_iter=250)
    
    data = {}

    # coefficient ci widths
    coefs = model.samples_['coefs']
    ci_mcmc = np.quantile(coefs, q=[0.025, 0.975], axis=0)
    ci_width_mcmc = np.abs(ci_mcmc[0, :] - ci_mcmc[1, :])
    data['coef1_mcmc'] = ci_width_mcmc[:, 0]
    data['coef2_mcmc'] = ci_width_mcmc[:, 1]

    coefs = (svi.samples_['W_coefs'] @ svi.B_fit_.todense()).transpose((0, 2, 1))
    ci_svi = np.quantile(coefs, q=[0.025, 0.975], axis=0)
    ci_width_svi = np.abs(ci_svi[0, :] - ci_svi[1, :])

    data['coef1_svi'] = ci_width_svi[:, 0]
    data['coef2_svi'] = ci_width_svi[:, 1]
    
    # density ci widths
    res = model.posterior_predictive(density)
    se_mcmc = np.quantile(res, q=[0.025, 0.975], axis=0)

    res = svi.posterior_predictive(density)
    se_svi = np.quantile(res, q=[0.025, 0.975], axis=0)

    ci_width_mcmc = np.abs(se_mcmc[0] - se_mcmc[1]).ravel()
    ci_width_svi = np.abs(se_svi[0] - se_svi[1]).ravel()

    data['density_mcmc'] = ci_width_mcmc
    data['density_svi'] = ci_width_svi

    # transitivity ci widths
    res = model.posterior_predictive(transitivity)
    se_mcmc = np.quantile(res, q=[0.025, 0.975], axis=0)

    res = svi.posterior_predictive(transitivity)
    se_svi = np.quantile(res, q=[0.025, 0.975], axis=0)

    ci_width_mcmc = np.abs(se_mcmc[0] - se_mcmc[1]).ravel()
    ci_width_svi = np.abs(se_svi[0] - se_svi[1]).ravel()

    data['transitivity_mcmc'] = ci_width_mcmc
    data['transitivity_svi'] = ci_width_svi

    # node #1 degree
    degree_func = functools.partial(node_degree, node_id=0)

    res = model.posterior_predictive(degree_func)
    se_mcmc = np.quantile(res, q=[0.025, 0.975], axis=0)

    res = svi.posterior_predictive(degree_func)
    se_svi = np.quantile(res, q=[0.025, 0.975], axis=0)

    ci_width_mcmc = np.abs(se_mcmc[0] - se_mcmc[1]).ravel()
    ci_width_svi = np.abs(se_svi[0] - se_svi[1]).ravel()

    data['degree_mcmc'] = ci_width_mcmc
    data['degree_svi'] = ci_width_svi
    
    data = pd.DataFrame(data)
    
    out_file = f'result_{seed}.csv'
    dir_base = 'output_mcmc_vi_comparison'
    dir_name = os.path.join(dir_base, f"sim_n{n_nodes}_T{n_time_points}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(os.path.join(dir_name, out_file), index=False)


for i in range(50):
    simulation(i)
