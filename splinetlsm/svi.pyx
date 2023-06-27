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
def test_me(m1, m2):
    #def test_me(a):
    #cdef np.ndarray[INDEX_T, ndim=1] indices = a.indices
    #cdef np.ndarray[INDEX_T, ndim=1] indptr = a.indptr
    #cdef np.ndarray[np.double_t, ndim=1] data = a.data
    #cdef np.int_t n_rows = a.shape[0]
    #cdef np.int_t n_cols = a.shape[1]
    #cdef sp_mat A = to_arma_csc(a.indices, a.indptr, a.data, a.shape[0], a.shape[1])
    #cdef sp_mat A = to_arma_csc(indices, indptr, data, n_rows, n_cols)
    cdef sp_mat A = to_arma_csc(m1)
    cdef sp_mat B = to_arma_csc(m2)
    cdef mat res 

    #cdef sp_mat B = to_arma_csc(sp_mat2)

    #return to_ndarray(res)
    #return sp_mat1.data
    print(A.n_rows)
    print(A.n_cols)
    print(A.n_nonzero)

    res = matrix_multiply(A, B)
    
    return to_ndarray(res)
  
def test_me2(m):
    cdef sp_mat X = speye(10, 10)
    #cdef mat A = to_arma_mat(m)
    cdef mat A = mat(X)
    
    return to_ndarray(A)
