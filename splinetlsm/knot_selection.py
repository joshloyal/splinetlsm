import copy
import numpy as np
import pandas as pd

from joblib import Parallel, delayed


def fit_model(base_model, Y, time_points, X, scale_factor, **fit_args):
    # copy model and set scale factor for knots
    model = copy.deepcopy(base_model)
    model.n_segments_scale_factor = scale_factor

    # fit the model
    model.fit(Y, time_points, X=X, **fit_args)

    return model, model.B_fit_.shape[0], model.waic()

def knot_selection(base_model, Y, time_points, X, scale_factors=[1, 2, 3, 4, 5], 
                   n_jobs=1, **fit_args):
    res = Parallel(n_jobs=n_jobs)(delayed(fit_model)(
        base_model=base_model, Y=Y, time_points=time_points, X=X, 
        scale_factor=s, **fit_args) for
            s in scale_factors)
    res = np.asarray(res)
    return list(res[:, 0]), pd.DataFrame(res[:, 1:], 
            columns=['spline_dimension', 'waic'])

