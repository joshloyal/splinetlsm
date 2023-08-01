# cython: language_level=3
"""Wrapper to convert between Armadillo structures and numpy arrays"""

import numpy as np
cimport numpy as np

from libcpp.string cimport string
from libcpp cimport bool


ctypedef fused INDEX_T:
    np.int32_t
    np.int64_t
    np.int_t
    np.uint32_t
    np.uint64_t
    int


cdef extern from "armadillo" namespace "arma" nogil:
    ctypedef unsigned int uword
    ctypedef int sword

    cdef struct arma_version:
        string as_string()

    cdef cppclass field[T]:
        field() nogil
        field(uword i) nogil

        T& operator()(const uword i) nogil
        #const T& operator()(const uword i) const

        int n_elem

    cdef cppclass cube:
        cube() nogil
        cube(double * aux_mem, int n_rows, int n_cols, int n_slices,
             bool copy_aux_mem, bool strict) nogil

        double * memptr() nogil

        int n_rows
        int n_cols
        int n_slices
        int n_elem

    cdef cppclass mat:
        # wrapper for arma::mat
        mat() nogil
        mat(double * aux_mem, int n_rows, int n_cols, bool copy_aux_mem,
            bool strict) nogil
        mat(sp_mat m) nogil

        double * memptr() nogil

        int n_rows
        int n_cols
        int n_elem

    cdef cppclass umat:
        # wrapper for arma::mat
        umat() nogil
        umat(uword * aux_mem, int n_rows, int n_cols, bool copy_aux_mem,
             bool strict) nogil

        uword * memptr() nogil

        int n_rows
        int n_cols
        int n_elem

    cdef cppclass vec:
        # wrapper for arma::vec
        vec() nogil
        vec(double * aux_mem, int number_of_elements, bool copy_aux_mem,
            bool strict) nogil

        double * memptr() nogil

        int n_elem

    cdef cppclass uvec:
        # wrapper for arma::vec
        uvec() nogil
        uvec(uword * aux_mem, int number_of_elements, bool copy_aux_mem,
             bool strict) nogil

        uword * memptr() nogil

        int n_elem

    cdef cppclass ivec:
        # wrapper for arma::vec
        ivec() nogil
        ivec(sword * aux_mem, int number_of_elements, bool copy_aux_mem,
             bool strict) nogil

        sword * memptr() nogil

        int n_elem

    cdef cppclass sp_mat:
        sp_mat() nogil
        sp_mat(uvec rowind, uvec colptr, vec values, int n_rows, int n_cols, bool check_for_zeros) nogil
        sp_mat(mat m)
        int n_rows
        int n_cols
        int n_elem
        int n_nonzero

    cdef sp_mat speye(int n_rows, int n_cols);

# convert to a numpy ndarray to an arma::mat
cdef mat to_arma_mat(np.ndarray[np.double_t, ndim=2] np_array, bool copy=*)

# convert from an arma::mat to a numpy ndarray
cdef np.ndarray[np.double_t, ndim=2] to_ndarray(const mat& arma_mat)

# convert from an arma::umat to a numpy ndarray
cdef np.ndarray[np.int_t, ndim=2] to_uint_ndarray(const umat& arma_mat)

# convert from a numpy ndarray to an arma::vec
cdef vec to_arma_vec(np.ndarray[np.double_t, ndim=1] np_array, bool copy=*)

# convert from an arma::vec to a numpy ndarray
cdef np.ndarray[np.double_t, ndim=1] to_1d_ndarray(const vec& arma_vec)

# convert from an arma::ivec to a numpy ndarray
cdef np.ndarray[np.int_t, ndim=1] to_1d_int_ndarray(const ivec& arma_vec)

# convert an numpy ndarray to an arma::vec
cdef uvec to_arma_uvec(np.ndarray[INDEX_T, ndim=1] np_array, bool copy=*)

# convert from an arma::uvec to a numpy ndarray
cdef np.ndarray[np.int_t, ndim=1] to_1d_uint_ndarray(const uvec& arma_vec)

# sparse matrices
cdef sp_mat to_arma_csc(m, bool copy=*)

# convert to a numpy ndarray to an arma::cube
cdef cube to_arma_cube(np.ndarray[np.double_t, ndim=3] np_array, bool copy=*)

# convert from an arma::cube to a numpy ndarray
cdef np.ndarray[np.double_t, ndim=3] to_3d_ndarray(const cube& arma_cube)
