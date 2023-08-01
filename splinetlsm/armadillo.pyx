# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np


def get_armadillo_version():
    cdef arma_version arma_ver

    return arma_ver.as_string()


cdef mat to_arma_mat(np.ndarray[np.double_t, ndim=2] np_array, bool copy=True):
    """Converts a numpy ndarray to an arma::mat. A copy of the array  is made
    if it either does not own its data or that data is stored in
    c-ordered format. Note that c-ordered is the default format
    for numpy arrays.
    """
    if np_array.flags.c_contiguous or not np_array.flags.owndata:
        np_array = np_array.copy(order='F')
    
    if copy:
        return mat(<double*>np_array.data,
                   np_array.shape[0], np_array.shape[1], True, False)
    else:
        return mat(<double*>np_array.data,
                   np_array.shape[0], np_array.shape[1], False, True)


cdef np.ndarray[np.double_t, ndim=2] to_ndarray(const mat& arma_mat):
    """Converts an arma::mat to a numpy ndarray. Currently, a
    copy is always made.
    """
    cdef int i = 0
    cdef n_rows = arma_mat.n_rows
    cdef n_cols = arma_mat.n_cols
    cdef const double* mat_ptr
    cdef double* np_ptr
    cdef np.ndarray[np.double_t, ndim=2] np_array

    # allocate memory for the new np.array
    np_array = np.empty((n_rows, n_cols),
                        dtype=np.double,
                        order="F")

    # copy data from the arma::mat to the numpy array
    mat_ptr = arma_mat.memptr()
    np_ptr = <double*> np_array.data
    for i in range(n_rows * n_cols):
        np_ptr[i] = mat_ptr[i]

    return np_array


cdef np.ndarray[np.int_t, ndim=2] to_uint_ndarray(const umat& arma_mat):
    """Converts an arma::mat to a numpy ndarray. Currently, a
    copy is always made.
    """
    cdef int i = 0
    cdef n_rows = arma_mat.n_rows
    cdef n_cols = arma_mat.n_cols
    cdef const uword* mat_ptr
    cdef np.int_t* np_ptr
    cdef np.ndarray[np.int_t, ndim=2] np_array

    # allocate memory for the new np.array
    np_array = np.empty((n_rows, n_cols),
                        dtype=int,
                        order="F")

    # copy data from the arma::mat to the numpy array
    mat_ptr = arma_mat.memptr()
    np_ptr = <np.int_t*> np_array.data
    for i in range(n_rows * n_cols):
        np_ptr[i] = mat_ptr[i]

    return np_array

cdef vec to_arma_vec(np.ndarray[np.double_t, ndim=1] np_array, bool copy=False):
    """Converts a 1d numpy array to an arma::vec. Data is copied if
    the array does not own its data.
    """
    if not np_array.flags.owndata:
        np_array = np_array.copy(order='F')
    
    if copy:
        return vec(<double*> np_array.data, np_array.shape[0], True, False)
    else:
        return vec(<double*> np_array.data, np_array.shape[0], False, True)


cdef np.ndarray[np.double_t, ndim=1] to_1d_ndarray(const vec& arma_vec):
    """Converts an arma::vec to a 1d numpy array. Currently, a copy is
    always made.
    """
    cdef int i = 0
    cdef n_elem = arma_vec.n_elem
    cdef const double* vec_ptr
    cdef double* np_ptr
    cdef np.ndarray[np.double_t, ndim=1] np_array

    # allocate memory for the new np.array
    np_array = np.empty(n_elem, dtype=np.double)

    # copy data from the arma::vec to the numpy array
    vec_ptr = arma_vec.memptr()
    np_ptr = <double*> np_array.data
    for i in range(n_elem):
        np_ptr[i] = <double>vec_ptr[i]

    return np_array


cdef np.ndarray[np.int_t, ndim=1] to_1d_int_ndarray(const ivec& arma_vec):
    """Converts an arma::vec to a 1d numpy array. Currently, a copy is
    always made.
    """
    cdef int i = 0
    cdef n_elem = arma_vec.n_elem
    cdef const sword* vec_ptr
    cdef np.int_t* np_ptr
    cdef np.ndarray[np.int_t, ndim=1] np_array

    # allocate memory for the new np.array
    np_array = np.empty(n_elem, dtype=int)

    # copy data from the arma::vec to the numpy array
    vec_ptr = arma_vec.memptr()
    np_ptr = <np.int_t*> np_array.data
    for i in range(n_elem):
        np_ptr[i] = <np.int_t>vec_ptr[i]

    return np_array


cdef uvec to_arma_uvec(np.ndarray[INDEX_T, ndim=1] np_array, bool copy=True):
    """Converts a 1d numpy array to an arma::uvec. Data is copied if
    the array does not own its data.
    """
    if not np_array.flags.owndata:
        np_array = np_array.copy(order='F')
    
    if copy:
        return uvec(<uword*> np_array.data, np_array.shape[0], True, False)
    else:
        return uvec(<uword*> np_array.data, np_array.shape[0], False, True)


cdef np.ndarray[np.int_t, ndim=1] to_1d_uint_ndarray(const uvec& arma_vec):
    """Converts an arma::vec to a 1d numpy array. Currently, a copy is
    always made.
    """
    cdef int i = 0
    cdef n_elem = arma_vec.n_elem
    cdef const uword* vec_ptr
    cdef np.int_t* np_ptr
    cdef np.ndarray[np.int_t, ndim=1] np_array

    # allocate memory for the new np.array
    np_array = np.empty(n_elem, dtype=int)

    # copy data from the arma::vec to the numpy array
    vec_ptr = arma_vec.memptr()
    np_ptr = <np.int_t*> np_array.data
    for i in range(n_elem):
        np_ptr[i] = <np.int_t>vec_ptr[i]

    return np_array


cdef sp_mat to_arma_csc(csc_mat, bool copy=True):
    # extract csc information
    # XXX: For some reason we need to convert to int64. Why?!
    cdef np.ndarray[np.int64_t, ndim=1] indices = csc_mat.indices.astype('int64')
    cdef np.ndarray[np.int64_t, ndim=1] indptr = csc_mat.indptr.astype('int64')
    cdef np.ndarray[np.double_t, ndim=1] data = csc_mat.data
    cdef int n_rows = csc_mat.shape[0]
    cdef int n_cols = csc_mat.shape[1]

    # convert to armadillo data structures
    cdef uvec rowind = to_arma_uvec(indices, copy=copy)
    cdef uvec colptr = to_arma_uvec(indptr, copy=copy)
    cdef vec values = to_arma_vec(data, copy=copy)

    return sp_mat(rowind, colptr, values, n_rows, n_cols, True)


cdef cube to_arma_cube(np.ndarray[np.double_t, ndim=3] np_array, bool copy=True):
    """Converts a numpy ndarray to an arma::cube. A copy of the array is made
    if it either does not own its data or that data is stored in
    c-ordered format. Note that c-ordered is the default format
    for numpy arrays.
    """
    if np_array.flags.c_contiguous or not np_array.flags.owndata:
        np_array = np_array.copy(order='F')
    
    if copy:
        return cube(<double*>np_array.data,
                   np_array.shape[0], np_array.shape[1], np_array.shape[2],
                   True, False)
    else:
        return cube(<double*>np_array.data,
                   np_array.shape[0], np_array.shape[1], np_array.shape[2],
                   False, True)


cdef np.ndarray[np.double_t, ndim=3] to_3d_ndarray(const cube& arma_cube):
    """Converts an arma::cube to a numpy ndarray. Currently, a
    copy is always made.
    """
    cdef int i = 0
    cdef n_rows = arma_cube.n_rows
    cdef n_cols = arma_cube.n_cols
    cdef n_slices = arma_cube.n_slices
    cdef const double* cube_ptr
    cdef double* np_ptr
    cdef np.ndarray[np.double_t, ndim=3] np_array

    # allocate memory for the new np.array
    np_array = np.empty((n_rows, n_cols, n_slices),
                        dtype=np.double,
                        order="F")

    # copy data from the arma::cube to the numpy array
    cube_ptr = arma_cube.memptr()
    np_ptr = <double*> np_array.data
    for i in range(n_rows * n_cols * n_slices):
        np_ptr[i] = cube_ptr[i]

    return np_array
