import numpy as np

from .bspline import bspline_basis
from ._svi import optimize_elbo_svi


class SplineDynamicLSM(object):
    def __init__(self,
                 n_features=10,
                 n_knots=10,
                 degree=3,
                 penalty_order=1,
                 coefs_penalty_order=1,
                 rate_prior=2.,
                 shape_prior=1.,
                 mgp_a1=2.,
                 mgp_a2=3.,
                 tau_prec=1e-2,
                 coefs_tau_prec=1e-2.
                 nonedge_proportion=2,
                 n_time_steps=n_time_steps,
                 step_size_delay=1,
                 step_size_power=0.75,
                 max_iter=500,
                 tol=1e-3,
                 random_state=42):
        self.n_features = n_features
        self.n_knots = n_knots
        self.degree = degree
        self.penalty_order = penalty_order
        self.coefs_penalty_order = coefs_penalty_order
        self.rate_prior = rate_prior
        self.shape_prior = shape_prior
        slef.mgp_a1 = mgp_a1
        self.mgp_a2 = mgp_a2
        self.tau_prec = tau_prec
        self.coefs_tau_prec = coefs_tau_prec
        self.nonedge_proportion = nonedge_proportion
        self.n_time_steps = n_time_steps
        self.step_size_delay = step_size_delay
        self.step_size_power = step_size_power
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, Y, time_points, X=None):
        
        n_time_steps = time_points.shape[0]
        n_nodes = Y[0].shape[0]

        if X is None:
            X = np.ones((n_time_steps, n_nodes, n_nodes, 1))

        self.time_points_ = time_points
        B, self.bs_ = bspline_basis(
                self.time_points_, n_knots=self.n_knots, degree=self.degree)
        
        self.params_, diagnostics = optimize_elbo_svi(
                y, B, X, time_points, 
                n_features=self.n_features,
                penalty_order=self.penalty_order,
                coefs_penalty_order=self.coefs_penalty_order,
                rate_prior=self.rate_prior, shape_prior=self.shape_prior,
                coefs_rate_prior=self.coefs_rate_prior, 
                coefs_shape_prior=self.coefs_shape_prior,
                mgp_a1=self.mgp_a1, mgp_a2=self.mgp_a2, 
                tau_prec=self.tau_prec, coefs_tau_prec=self.coefs_tau_prec,
                nonedge_proportion=self.nonedge_proportion, 
                n_time_steps=self.n_time_steps, 
                step_size_delay=self.step_size_delay,
                step_size_power=self.step_size_power, max_iter=self,max_iter,
                tol=self.tol, random_state=self.random_state)

        self.converged_ = diagnostics['converged']
        self.n_iter_ = diagnostics['n_iter']
        self.diffs_ = diagnostics['diffs']

        return self
