# cython: language_level=3


import numpy as np
cimport numpy as np

from splinetlsm.armadillo cimport mat, sp_mat, speye, INDEX_T, uvec
from splinetlsm.armadillo cimport to_ndarray, to_arma_csc, to_arma_mat, to_1d_uint_ndarray

cdef extern from "splinetlsm.h" namespace "splinetlsm" nogil:
    cdef uvec randperm(unsigned int N, unsigned int M)
