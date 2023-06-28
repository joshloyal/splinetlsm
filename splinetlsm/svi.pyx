# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


import numpy as np
cimport numpy as np
np.import_array()

#def test_me(np.ndarray[INDEX_T, ndim=1] indices,
#            np.ndarray[INDEX_T, ndim=1] indptr,
#            np.ndarray[np.double_t, ndim=1] data,
#            int n_rows, int n_cols):
def test_me(N, M):
    cdef uvec res 

    #cdef sp_ma_multiply(A, B)
    res = randperm(N, M)
    
    return to_1d_uint_ndarray(res)
  
def test_me2(m):
    cdef sp_mat X = speye(10, 10)
    #cdef mat A = to_arma_mat(m)
    cdef mat A = mat(X)
    
    return to_ndarray(A)
