import numpy as np
import statsmodels.api as sm

from .initialize import smooth_positions_procrustes, logit_svt, ase
from .static_gof import vec_to_adjacency
from polyagamma import random_polyagamma
from tqdm import tqdm
from sklearn.utils import check_random_state
from sklearn.metrics import roc_auc_score
from scipy.linalg import block_diag, orthogonal_procrustes
from scipy.special import expit
from scipy import stats 


def dynamic_adjacency_to_vec(Y):
    n_time_points, n_nodes, _ = Y.shape
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    subdiag = np.tril_indices(n_nodes, k=-1)
    y = np.zeros((n_time_points, n_dyads), dtype=int)
    for t in range(n_time_points):
        y[t] = Y[t][subdiag]

    return y


def calculate_auc(Y, probas):
    n_time_steps, _ = Y.shape
    y_true, y_pred = [], []
    for t in range(n_time_steps):
        y_true.append(Y[t])
    
    return roc_auc_score(np.concatenate(y_true), probas.ravel())


def initialize_parameters(Y, n_features=2, random_state=42): 
    n_time_steps, n_nodes, _ = Y.shape
    n_nodes = Y[0].shape[0]
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    
    # estimate log odds
    theta = np.stack([logit_svt(Y[t], random_state=random_state) for 
        t in range(n_time_steps)], axis=0)
    
    # estimate coefs and latent positions using OLS and ASE
    intercept = np.zeros(n_time_steps)
    if n_features is not None:
        U = np.zeros((n_time_steps, n_nodes, n_features))
    else:
        U = None

    for t in range(n_time_steps):
        intercept[t] = np.mean(theta[t])
        if n_features is not None:
            resid = np.asarray(vec_to_adjacency(theta[t] - intercept[t]))

            # clip residual to prevent large latent position values
            resid = np.clip(resid, -5, 5)
             
            # ASE of residual at time t
            U[t] = ase(resid, k=n_features)
     
    # smooth the latent positions
    if n_features is not None:
        U = smooth_positions_procrustes(U)
        U = U.transpose((1, 2, 0))
    
    return intercept, None


class LatentPositionNestedGP(sm.tsa.statespace.MLEModel):
    """
        y_t = Z_t alpha_t + epsilon_t,  epsilon_t ~ N(0, H_t)
        alpha_{t+1} = T_t alpha_t + R_t eta_t,  eta_t ~ N(0, Q_t)    
    """
    def __init__(self, y, X):
        # y : (nobs, k_endog)

        # time-varying observation covariance matrix
        obs_cov = np.zeros((y.shape[1], y.shape[1], y.shape[0]))
        self.n_features = X.shape[1]
        k_states = 3 * self.n_features
        k_posdef = 2 * self.n_features
        super(LatentPositionNestedGP, self).__init__(endog=y, 
              k_states=k_states, k_posdef=k_posdef,
              obs_cov=obs_cov, initialization='diffuse')
         
        self.ssm['design'] = np.zeros((X.shape[0], 3 * X.shape[1], X.shape[2]))
        self.ssm['design'][:, ::3, :] = X

        R = np.array([[0., 0.],
                      [1., 0.],
                      [0., 1.]])       # R_t
        self.ssm['selection'] = block_diag(*(self.n_features * [R]))

        T = np.array([[1., 1., 0.],
                      [0., 1., 1.],
                      [0., 0., 1.]])  # T_t
        self.ssm['transition'] = block_diag(*(self.n_features * [T]))
      
    @property
    def start_params(self):
        return np.ones(2 * self.n_features + self.k_endog * self.nobs)

    def update(self, params, **kwargs):
        params = super(LatentPositionNestedGP, self).update(params, **kwargs)
        self['state_cov'] = np.diag(params[:(2 * self.n_features)])
 
        obs_vars = params[(2 * self.n_features):]
        for t in range(self.nobs):
            idxs = slice(self.k_endog * t, self.k_endog * (t+1))
            self['obs_cov', :, :, t] = np.diag(obs_vars[idxs])


class InterceptNestedGP(sm.tsa.statespace.MLEModel):
    """
        y_t = Z_t alpha_t + epsilon_t,  epsilon_t ~ N(0, H_t)
        alpha_{t+1} = T_t alpha_t + R_t eta_t,  eta_t ~ N(0, Q_t)    
    """
    def __init__(self, y):
        # time-varying observation covariance matrix
        obs_cov = np.zeros((1, 1, y.shape[0]))
        obs_cov[0, 0] = 1.
        super(InterceptNestedGP, self).__init__(endog=y, exog=None, k_states=3, k_posdef=2,
              obs_cov=obs_cov, initialization='diffuse')

        self.ssm['design'] = np.array([1., 0., 0.])        # Z_t
        self.ssm['selection'] = np.array([[0., 0.],
                                          [1., 0.],
                                          [0., 1.]])       # R_t
        self.ssm['transition'] = np.array([[1., 1., 0.],
                                           [0., 1., 1.],
                                           [0., 0., 1.]])  # T_t
         
    @property
    def start_params(self):
        return np.ones(2 + self.nobs)

    def update(self, params, **kwargs):
        params = super(InterceptNestedGP, self).update(params, **kwargs)
        self['state_cov'] = np.diag(params[:2])
        self['obs_cov', 0, 0] = params[2:]


class LadyNetworkModel(object):
    def __init__(self,
                 n_features=2,
                 prior_variance_shape=0.01,
                 prior_variance_scale=0.01,
                 random_state=42):
        self.n_features = n_features
        self.prior_variance_shape = prior_variance_shape
        self.prior_variance_scale = prior_variance_scale
        self.random_state = random_state

    def sample(self, Y, n_warmup=2500, n_samples=2500):
        n_time_points, n_nodes, _ = Y.shape
        n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
        self.Y_fit_ = Y.copy()
        self.y_vec_ = dynamic_adjacency_to_vec(Y)
        rng = check_random_state(self.random_state)
        
        # initialize samples
        self.samples_ = {
                'intercept': np.zeros((n_samples, n_time_points)),
                'sigma_intercept': np.zeros((n_samples, 2)),
        }
        if self.n_features is not None:
            self.samples_['X'] = np.zeros(
                    (n_samples, n_time_points, n_nodes, self.n_features))
        
        # initialize parameters
        subdiag = np.tril_indices(n_nodes, k=-1)
        self.probas_ = np.zeros((n_time_points, n_dyads))
        omega = np.zeros((n_time_points, n_nodes, n_nodes))
        dist = np.zeros((n_time_points, n_dyads))
        #dist = np.zeros((n_time_points, n_nodes, n_nodes))
        
        #intercept, X = initialize_parameters(Y, n_features=self.n_features)
        intercept = rng.randn(n_time_points)
        sigma_intercept = np.ones(2)
        
        if self.n_features is not None:
            X = rng.randn(n_nodes, self.n_features, n_time_points)
            sigma_X = np.ones((n_nodes, 2 * self.n_features))
            #dist = np.einsum('tih,tjh->tij', X, X)
            for t in range(n_time_points):
                dist[t] = (X[..., t] @ X[..., t].T)[subdiag]
 
        kappa = self.y_vec_ - 0.5
        for idx in tqdm(range(n_warmup + n_samples)):
            # sample polya-gamma latent variables
            for t in range(n_time_points):
                if self.n_features is not None:
                    omega[t][subdiag] = random_polyagamma(
                        z=intercept[t] + dist[t], size=n_dyads, 
                        random_state=t + idx)
                else:
                    omega[t][subdiag] = random_polyagamma(
                        z=intercept[t], size=n_dyads, random_state=t + idx)
                omega[t].T[subdiag] = omega[t][subdiag]
            
            # sample intercept
            r = np.zeros(n_time_points)
            obs_var = np.zeros(n_time_points)
            for t in range(n_time_points):
                if self.n_features is not None:
                    r[t] = np.sum(kappa[t] - omega[t][subdiag] * dist[t])
                else:
                    r[t] = np.sum(kappa[t])
                
                obs_var[t] = 1. / np.sum(omega[t][subdiag])
            
            intercept_nGP = InterceptNestedGP(r * obs_var)
            intercept_smoother = intercept_nGP.simulation_smoother()
            
            params = np.r_[sigma_intercept, obs_var]
            intercept_nGP.update(params)
            intercept_smoother.simulate()

            intercept = intercept_smoother.simulated_state[0]
            v = intercept_smoother.simulated_state[1]
            z = intercept_smoother.simulated_state[2]
    
            # sample intercept variances
            shape = self.prior_variance_shape + 0.5 * (n_time_points - 1)
            
            scale = self.prior_variance_scale + 0.5 * np.sum(
                    (v[1:] - v[:-1] - z[:-1]) ** 2) 
            sigma_intercept[0] = stats.invgamma.rvs(shape, scale=scale, random_state=rng)
            
            scale = self.prior_variance_scale + 0.5 * np.sum(np.diff(z) ** 2) 
            sigma_intercept[1] = stats.invgamma.rvs(shape, scale=scale, random_state=rng)

            # sample latent positions
            if self.n_features is not None:
                for i in range(n_nodes):
                    node_mask = ~np.isin(np.arange(n_nodes), i)
                    
                    obs_var = 1. / omega[:, i, node_mask]
                     
                    # design matrix
                    X_v = X[node_mask]

                    # observations
                    y_v = obs_var * (self.Y_fit_[:, i, node_mask] - 0.5) - intercept.reshape(-1, 1)
                    
                    Xv_nGP = LatentPositionNestedGP(y_v, X_v)
                    Xv_smoother = Xv_nGP.simulation_smoother()
                    
                    # obs_var ravel is [1./omega_v(t_1) | 1./omega_v(t_2) | ... ]
                    params = np.r_[sigma_X[i], obs_var.ravel()]
                    Xv_nGP.update(params)
                    Xv_smoother.simulate()

                    X[i] = Xv_smoother.simulated_state[::3]
                    v = Xv_smoother.simulated_state[1::3].T  # T x n_features
                    z = Xv_smoother.simulated_state[2::3].T
                    
                    # sample latent position variances
                    shape = self.prior_variance_shape + 0.5 * (n_time_points - 1)
                    
                    scale = self.prior_variance_scale + 0.5 * np.sum((v[1:] - v[:-1] - z[:-1]) ** 2, axis=0)
                    sigma_X[i, ::2] = stats.invgamma.rvs(shape, scale=scale, random_state=rng)
                    
                    scale = self.prior_variance_scale + 0.5 * np.sum(np.diff(z, axis=0) ** 2, axis=0)
                    sigma_X[i, 1::2] = stats.invgamma.rvs(shape, scale=scale, random_state=rng)
            
            # update distances
            if self.n_features is not None:
                for t in range(n_time_points):
                    dist[t] = (X[..., t] @ X[..., t].T)[subdiag]

            if idx >= n_warmup:
                self.samples_['intercept'][idx-n_warmup] = intercept
                self.samples_['sigma_intercept'][idx-n_warmup] = sigma_intercept
                if self.n_features is not None:
                    self.samples_['X'][idx-n_warmup] = X.transpose((2, 0, 1))
                self.probas_ += expit(intercept.reshape(-1, 1) + dist) / n_samples

        # post processing
        self.intercept_ = self.samples_['intercept'].mean(axis=0)
        if self.n_features is not None:
            for idx in range(n_samples):
                for t in range(n_time_points):
                    R, _ = orthogonal_procrustes(
                        self.samples_['X'][idx][t], self.samples_['X'][-1][t])
                    self.samples_['X'][idx][t] = self.samples_['X'][idx][t] @ R
            self.X_ = smooth_positions_procrustes(self.samples_['X'].mean(axis=0))

        self.auc_ = calculate_auc(self.y_vec_, self.probas_)

        return self
