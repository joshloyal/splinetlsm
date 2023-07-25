import numpy as np
import scipy.sparse as sp

from scipy.interpolate import BSpline


def bspline_basis(time_points, n_segments=10, n_knots=None, degree=3, return_knots=False, 
                  return_sparse=True): 
    # XXX: For efficient slicing in csc format the design matrix is stored
    # with dimensions L_M x n_time_points
    n_segments = n_knots - 1 if n_knots is not None else n_segments

    dx = 1 / n_segments
    x_low = np.min(time_points)
    x_high = np.max(time_points)
    dx = (x_high - x_low) / n_segments

    knots = np.arange(x_low - degree * dx, 1 + (degree + 1) * dx, step = dx)
    bs = BSpline(knots, np.eye(knots.shape[0]), k=degree)
    
    if return_sparse:
        B = sp.csc_matrix(bs.design_matrix(time_points, knots, k=degree).T)
    else:
        B = bs(time_points).T

    if return_knots:
        #return bs(time_points)[:, degree:-degree], bs, knots[degree:-degree]
        return B, bs, knots[degree:-degree]

    return B, bs
    
