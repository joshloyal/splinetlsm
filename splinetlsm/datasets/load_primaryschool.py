import joblib
import numpy as np
import pandas as pd

from os.path import dirname, join


__all__ = ['load_primaryschool']


def load_primaryschool(reference_layer='Thursday'):
    module_path = dirname(__file__)
    file_path = join(module_path, 'raw_data')

    file_name = join(file_path, 'sp_primary_school', 
                     'primaryschool10.gz')
    Y = joblib.load(open(file_name, 'rb'))

    # covariates
    file_name = join(file_path, 'sp_primary_school', 
                     'covariates.csv')
    X = pd.read_csv(file_name).values

    time_points = np.arange(Y.shape[0])
    time_points = time_points / time_points[-1]

    # labels
    #layer_labels = ['Thursday', 'Friday']

    #time_labels = [
    #    "8:30 to 9:20", "9:20 to 9:40", "9:40 to 10:00", "10:00 to 10:20",
    #    "10:20 to 10:40", "10:40 to 11:00", "11:00 to 11:20", "11:20 to 11:40",
    #    "11:40 to 12:00", "12:00 to 12:20", "12:20 to 12:40", "12:40 to 1:00",
    #    "1:00 to 1:20", "1:20 to 1:40", "1:40 to 2:00", "2:00 to 2:20",
    #    "2:20 to 2:40", "2:40 to 3:00", "3:00 to 3:20", "3:20 to 3:40",
    #    "3:40 to 4:00", "4:00 to 4:20", "4:20 to 4:40", "4:40 to 5:30"
    #]

    #if reference_layer != 'Thursday':
    #    Y = Y[::-1]
    #    layer_labels = ['Friday', 'Thursday']

    return np.ascontiguousarray(Y), time_points, X#, layer_labels, time_labels
