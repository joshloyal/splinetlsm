import joblib
import numpy as np
import pandas as pd

from datetime import datetime as dt
from os.path import dirname, join

from ..mcmc import dynamic_adjacency_to_vec


__all__ = ['load_polecat']


def load_polecat(n_nodes=None, n_time_points=None):
    module_path = dirname(__file__)
    file_path = join(module_path, 'raw_data')

    file_name = join(file_path, 'polecat', 
                     'polecat_weekly_matcon_subset.gz')
    Y_raw = np.ascontiguousarray(joblib.load(open(file_name, 'rb')))
    Y = Y_raw[1:]  # set up for lagged covariates

    # node names
    node_names = pd.read_csv(
            join(file_path, 'polecat', 'dist_node_names.npy'), 
            header=None).values.ravel()
 
    #iso_codes = pd.read_csv(join(file_path, 'polecat', 'iso_codes.csv'))
    iso_codes = pd.read_csv(join(file_path, 'polecat', 'iso_regions.csv'))
    iso_map = {row['Country']: row['ISO3'] for (idx, row) in iso_codes.iterrows()} 
    region_map = {row['Country']: row['Region'] for (idx, row) in iso_codes.iterrows()} 
    iso_codes = np.array([iso_map[i] for i in node_names])
    regions = np.array([region_map[i] for i in node_names])

    # load static covariates (contig, comlang_off, distw)
    n_time_steps = Y.shape[0]
    file_name = join(file_path, 'polecat', 'dist_covariates.gz')
    X_static = np.ascontiguousarray(joblib.load(open(file_name, 'rb')))[None, ...]
    X_static = np.repeat(X_static, n_time_steps, axis=0)
    X_static[..., 2] = np.log1p(X_static[..., 2])
    # remove contig (to much colinearity with distance)
    X_static = X_static[..., [1, 2]]

    file_name = join(file_path, 'polecat', 'polecat_weekly_covariates_subset.gz')
    X = np.ascontiguousarray(joblib.load(open(file_name, 'rb')))
    X = X[:-1] # lagged covariates

    ## Multicollinearity is strong in the raw covariates. 
    ## Let use the excess of material cooperation over verbal coop
    X = np.stack((Y_raw[:-1], X[..., 2] - X[..., 1] - X[..., 0]), axis=-1)
    X = np.concatenate((X, X_static), axis=-1)
    #X = np.expand_dims(X[..., 2] - X[..., 1] - X[..., 0], axis=-1)

    # standardize covariates by unconditional std
    for k in [1, 3]:
        x_dyn = dynamic_adjacency_to_vec(X[..., k])
        x_std = x_dyn.std()
        x_mean = x_dyn.mean()
        X[..., k] -= x_mean
        X[..., k] /=x_std 

   
    time_points = np.arange(Y.shape[0])
    time_labels = []
    years = [2018, 2019, 2020, 2021, 2022]
    for t in np.arange(Y.shape[0]+1):
        week = t % 52 + 1
        year = years[t // 52] 
        date = dt.fromisocalendar(year, week, 1)
        time_labels.append(date.strftime("%Y-%m-%W"))
    
    # limit to n_nodes with highest overall degree
    if n_nodes is not None:
        degree = Y.sum(axis=(0, 1))
        order = np.argsort(degree)[::-1][:n_nodes]
        Y = Y[:, order][..., order]
        X = X[:, order][:, :, order, :]
        node_names = node_names[order]
        iso_codes = iso_codes[order]
        regions = regions[order]
    
    time_labels = time_labels[1:]
    if n_time_points is not None:
        Y = Y[-n_time_points:]
        X = X[-n_time_points:]
        time_points = time_points[-n_time_points:]
        time_labels = time_labels[-n_time_points:]
    
    return Y, time_points, X, node_names, iso_codes, regions, time_labels
