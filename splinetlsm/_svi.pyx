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

cdef np.ndarray to_4d_ndarray(field[cube] arma_array):
    cdef size_t n_elem = arma_array.n_elem
    cdef list out = []
    cdef cube A

    for t in range(n_elem):
        A = arma_array(t)
        out.append(to_3d_ndarray(A))

    return np.asarray(out)


def optimize_elbo_svi(Y, B, time_points, X,
        W_init, W_coefs_init,
        n_features=2, penalty_order=1, coefs_penalty_order=2,
        rate_prior=2., shape_prior=1., 
        coefs_rate_prior=2., coefs_shape_prior=1.,
        mgp_a1=2., mgp_a2=3., 
        tau_prec=0.01, coefs_tau_prec=0.01,
        nonedge_proportion=5, uint n_time_steps=5,
        step_size_delay=1., step_size_power=0.75,
        max_iter=100, tol=0.001, random_state=42):
    
    cdef uint n_time_points = time_points.shape[0]
    cdef field[sp_mat] Y_arma = to_sparse_cube(Y)
    cdef sp_mat B_arma = to_arma_csc(B)
    cdef field[cube] X_arma
    cdef cube W_init_arma = to_arma_cube(W_init)
    cdef mat W_coefs_init_arma = to_arma_mat(W_coefs_init)
    cdef vec time_points_arma = to_arma_vec(time_points)

    cdef ModelParams params
    cdef Moments mom

    cdef np.ndarray[np.double_t, ndim=2] W_coefs
    cdef np.ndarray[np.double_t, ndim=3] W_coefs_sigma
    cdef np.ndarray[np.double_t, ndim=1] b_coefs
    cdef np.ndarray[np.double_t, ndim=2] coefs
    cdef np.ndarray[np.double_t, ndim=1] w_coefs_prec
    
    if X is None:
        X_arma = field[cube](n_time_points)
    else:
        X_arma = to_arma_4d(X)

    result = optimize_elbo(
        Y_arma, B_arma, X_arma, time_points_arma,
        W_init_arma, W_coefs_init_arma,
        n_features=n_features, 
        penalty_order=penalty_order, coefs_penalty_order=coefs_penalty_order,
        rate_prior=rate_prior, shape_prior=shape_prior, 
        coefs_rate_prior=coefs_rate_prior, coefs_shape_prior=coefs_shape_prior,
        mgp_a1=mgp_a1, mgp_a2=mgp_a2, 
        tau_prec=tau_prec, coefs_tau_prec=coefs_tau_prec,
        nonedge_proportion=nonedge_proportion, n_time_steps=n_time_steps,
        step_size_delay=step_size_delay, step_size_power=step_size_power,
        max_iter=max_iter, tol=tol, random_state=random_state)
    
    W_coefs = to_ndarray(result.params.W_coefs)
    W_coefs_sigma = to_3d_ndarray(result.params.W_coefs_sigma)
    b_coefs = to_1d_ndarray(result.params.b_coefs)

    parameters = {
        'W': to_3d_ndarray(result.params.W),                 # n x L_m x d
        'W_sigma': to_4d_ndarray(result.params.W_sigma),     # n x L_m x L_m x d
        'W_intercept': W_coefs[:, 0],                        # (L_m,)
        'W_intercept_sigma': W_coefs_sigma[..., 0],          # L_m x L_m
        'W_coefs': W_coefs[..., 1:],                         # L_m x p
        'W_coefs_sigma': W_coefs_sigma[..., 1:],             # L_m x L_m x p
        'b': to_1d_ndarray(result.params.b),                 # (n,)
        'b_intercept': b_coefs[0],
        'b_coefs': b_coefs[1:],                              # (p,)
        'mgp_rate': to_1d_ndarray(result.params.mgp_rate),   # (d,)
        'mgp_shape': to_1d_ndarray(result.params.mgp_shape), # (d,)
        'a': result.params.a,
        'p': result.params.p,
        'a_coefs': result.params.a_coefs,
        'p_coefs': result.params.p_coefs
    } 

    diagnostics = {
        'converged': result.converged,
        'n_iter': result.n_iter,
        'diffs': to_1d_ndarray(result.parameter_difference),
        'loglik': to_1d_ndarray(result.loglik),
        'step_size': to_1d_ndarray(result.step_size)
    }

    mom = calculate_moments(result.params, B_arma)
    coefs = to_ndarray(mom.coefs)
    w_coefs_prec = to_1d_ndarray(mom.w_coefs_prec)
    moments = {
            'U': to_3d_ndarray(mom.U),                       # n x T x d
            'intercept': coefs[0],                           # (T,)
            'coefs': coefs[1:],                              # p x T
            'w_prec': to_1d_ndarray(mom.w_prec),             # (n,)
            'w_intercept_prec': w_coefs_prec[0],              
            'w_coefs_prec': w_coefs_prec[1:],                # (p,)
            'gamma': np.exp(to_1d_ndarray(mom.log_gamma))    # (d,)

    }

    return parameters, moments, diagnostics
