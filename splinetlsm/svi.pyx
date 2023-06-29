# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


import numpy as np
cimport numpy as np
np.import_array()


cdef field[sp_mat] to_sparse_cube(list Y):
    cdef size_t n_time_steps = len(Y)
    cdef size_t t = 0
    cdef field[sp_mat] Y_sp = field[sp_mat](n_time_steps)
    cdef sp_mat Y_t
    for t in range(n_time_steps):
        Y_t = to_arma_csc(Y[t])
        set_spcube_value(Y_sp, Y_t, t)

    return Y_sp 


cdef field[cube] to_arma_4d(np.ndarray[np.double_t, ndim=4] np_array):
    cdef size_t n_time_steps = np_array.shape[0]
    cdef size_t t = 0
    cdef cube X_t
    cdef field[cube] X = field[cube](n_time_steps)

    for t in range(n_time_steps):
        X_t = to_arma_cube(np_array[t])
        set_4darray_value(X, X_t, t)
    
    return X


def optimize_elbo(Y, X, B, time_points):
    cdef field[sp_mat] Y_arma = to_sparse_cube(Y)
    cdef field[cube] X_arma = to_arma_4d(X)
    cdef sp_mat B_arma = to_arma_csc(B)
    #cdef vec time_points_arma = to_arma_vec(time_points)

