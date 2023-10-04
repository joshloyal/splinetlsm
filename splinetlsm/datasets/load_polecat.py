import joblib
import numpy as np
import pandas as pd

from datetime import datetime as dt
from os.path import dirname, join

from ..mcmc import dynamic_adjacency_to_vec


__all__ = ['load_polecat']


def load_polecat():
    module_path = dirname(__file__)
    file_path = join(module_path, 'raw_data')

    file_name = join(file_path, 'polecat', 
                     'polecat_weekly_matcon.gz')
    Y_raw = np.ascontiguousarray(joblib.load(open(file_name, 'rb')))
    Y = Y_raw[1:]  # set up for lagged covariates

    file_name = join(file_path, 'polecat', 
                     'polecat_weekly_covariates.gz')
    X = np.ascontiguousarray(joblib.load(open(file_name, 'rb')))
    X = X[:-1] # lagged covariates

    # Multicollinearity is strong in the raw covariates. 
    # Let use the excess of material cooperation over verbal coop
    X = np.stack((Y_raw[:-1], X[..., 2] - X[..., 1] - X[..., 0]), axis=-1)
    #X = np.expand_dims(X[..., 2] - X[..., 1] - X[..., 0], axis=-1)
    
    # standardize covariates by unconditional std
    for k in range(1, X.shape[-1]):
        x_std = dynamic_adjacency_to_vec(X[..., k]).std()
        X[..., k] /= x_std 

    node_names = pd.read_csv(
            join(file_path, 'polecat', 'node_names.npy'), 
            header=None).values.ravel()
   
    time_points = np.arange(Y.shape[0])
    time_labels = []
    years = [2018, 2019, 2020, 2021, 2022]
    for t in np.arange(Y.shape[0]+1):
        week = t % 52 + 1
        year = years[t // 52] 
        date = dt.fromisocalendar(year, week, 1)
        time_labels.append(date.strftime("%Y-%m-%W"))


    return Y, time_points, X, node_names, time_labels[1:]
