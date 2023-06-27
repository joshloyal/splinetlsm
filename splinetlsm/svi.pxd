# cython: language_level=3


import numpy as np
cimport numpy as np

from splinetlsm.armadillo cimport mat, sp_mat, speye, INDEX_T
from splinetlsm.armadillo cimport to_ndarray, to_arma_csc, to_arma_mat

cdef extern from "splinetlsm.h" namespace "splinetlsm" nogil:
    cdef mat matrix_multiply(sp_mat& X, sp_mat& Y)
