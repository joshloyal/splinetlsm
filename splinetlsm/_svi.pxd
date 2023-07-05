# cython: language_level=3


import numpy as np
cimport numpy as np

from libcpp cimport bool

from splinetlsm.armadillo cimport mat, sp_mat, field, cube, vec
from splinetlsm.armadillo cimport to_arma_csc, to_arma_cube, to_arma_vec
from splinetlsm.armadillo cimport to_ndarray, to_1d_ndarray, to_3d_ndarray


cdef extern from "splinetlsm.h" namespace "splinetlsm" nogil:

    cdef cppclass sp_cube
    cdef cppclass array4d
    ctypedef unsigned int uint

    cdef cppclass ModelParams:
        cube W
        field[cube] W_sigma
        
        mat W_coefs
        cube W_coefs_sigma

        vec b
        vec b_coefs
        vec mgp_rate

        double a
        double p

        double a_coefs
        double p_coefs

        vec mgp_shape
    
    cdef struct Moments:
        cube U
        cube U_sigma
        
        mat coefs
        mat coefs_sigma
        
        vec w_prec
        vec w_coefs_prec

        vec log_gamma
        
    cdef cppclass SVIResult:
        ModelParams params
        bool converged
        vec parameter_difference
        vec loglik
        uint n_iter

    cdef void set_spcube_value(field[sp_mat]& Y, sp_mat& B, uint index) 
    cdef void set_4darray_value(field[cube]& Y, cube& B, uint index)
    
    cdef Moments calculate_moments(const ModelParams& params, const sp_mat& B)

    cdef SVIResult optimize_elbo(
        const field[sp_mat]& Y, 
        const sp_mat& B, const field[cube]& X, 
        const vec& time_points, 
        uint n_features, 
        uint penalty_order, uint coefs_penalty_order,
        double rate_prior, double shape_prior, 
        double coefs_rate_prior, double coefs_shape_prior, 
        double mgp_a1, double mgp_a2, 
        double tau_prec, double coefs_tau_prec, 
        double nonedge_proportion, uint n_time_steps, 
        double step_size_delay, double step_size_power, 
        uint max_iter, double tol, 
        int random_state)
