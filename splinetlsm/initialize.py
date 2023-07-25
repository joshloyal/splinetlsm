import numpy as np
import scipy.sparse as sp

from scipy.special import logit
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import LinearRegression

from .static_gof import vec_to_adjacency
from .mcmc import dynamic_adjacency_to_vec


#EPS = np.finfo('float64').epsneg
EPS = 1e-3

def smooth_positions_procrustes(U):
    n_time_steps, _, _ = U.shape
    for t in range(1, n_time_steps):
        R, _ = orthogonal_procrustes(U[t], U[t-1])
        U[t] = U[t] @ R

    return U


def covariates_to_vec(X):
    n_nodes, _, n_covariates = X.shape
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    subdiag = np.tril_indices(n_nodes, k=-1)

    x = np.zeros((n_dyads, n_covariates))
    for k in range(n_covariates):
        x[:, k] = X[..., k][subdiag]
    
    return x


def logit_svt(Y, random_state=42):
    n_nodes = Y.shape[0]
    subdiag = np.tril_indices(n_nodes, k=-1)

    tau = np.sqrt(Y.mean())
    u,s,v = sp.linalg.svds(Y, k=n_nodes-1, random_state=random_state)
    #u,s,v = np.linalg.svd(Y.toarray())
    ids = s >= tau
    p_tilde = np.clip(u[:, ids] @ np.diag(s[ids]) @ v[ids, :], 1e-2, 0.5)#EPS, 1-EPS)

    return logit(0.5 * (p_tilde + p_tilde.T))[subdiag]



#def init_intercept(theta, B):
#    n_time_steps, n_dyads = theta.shape
#
#    X = sp.csr_matrix(np.repeat(B.T.todense(), n_dyads, axis=0))
#    y = theta.ravel()
#
#    reg = LinearRegression(fit_intercept=False)
#    reg.fit(X, y)
#    
#    resid = y - reg.predict(X)
#
#    return reg.coef_, resid.reshape(n_time_steps, n_dyads)
#
#
#def init_covariate(theta, Z, B):
#    n_time_steps, n_dyads = theta.shape
#    
#    z_vec = dynamic_adjacency_to_vec(Z).ravel().reshape(-1, 1)
#    X = sp.csr_array(z_vec * np.repeat(B.T.todense(), n_dyads, axis=0))
#    y = theta.ravel()
#
#    reg = LinearRegression(fit_intercept=False)
#    reg.fit(X, y)
#    
#    resid = y - reg.predict(X)
#
#    return reg.coef_, resid.reshape(n_time_steps, n_dyads)


def init_covariates(theta, X):
    x = covariates_to_vec(X)
    y = theta.ravel()
    reg = LinearRegression()
    reg.fit(x, y)
    resid = y - reg.predict(x)

    return np.r_[reg.intercept_, reg.coef_], np.asarray(vec_to_adjacency(resid))


def initialize_parameters(Y, B, X=None, n_features=2, random_state=42): 
    n_time_steps = len(Y)
    n_nodes = Y[0].shape[0]
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))

    # estimate log odds
    theta = np.stack([logit_svt(Y[t], random_state=random_state) for 
        t in range(n_time_steps)], axis=0)
    n_covariates = 1 if X is None else 1 + X.shape[-1]
    W_coefs = np.zeros((B.shape[0], n_covariates))

    #W_coefs[:, 0], resid = init_intercept(theta, B)  
    #if X is not None:
    #    for k in range(X.shape[-1]):
    #        W_coefs[:, k+1], resid = init_covariate(resid, X[..., k], B)
    
    coefs = np.zeros((n_time_steps, n_covariates))
    U = np.zeros((n_time_steps, n_nodes, n_features))
    for t in range(n_time_steps):
        if X is None:
            coefs[t] = np.mean(theta[t])
            resid = np.asarray(vec_to_adjacency(theta[t] - coefs[t]))
        else:
            coefs[t], resid = init_covariates(theta[t], X[t])
        
        resid = np.clip(resid, -5, 5)
        
        #u, s, v = sp.linalg.svds(resid, k=n_features)
        #U[t] = u * np.sqrt(s)
        
        eigvals, eigvec = np.linalg.eigh(resid)
        order = np.argsort(np.abs(eigvals))[::-1][:n_features]
        U[t] = eigvec[:, order] * np.sqrt(np.abs(eigvals[order])) 

    
    # smooth the latent positions
    U = smooth_positions_procrustes(U)
    
    # project onto spline coefficients
    H = np.linalg.pinv((B @ B.T).toarray()) @ B
    W_coefs = H @ coefs 
    W = np.zeros((B.shape[0], n_nodes, n_features))
    for d in range(n_features):
        W[..., d] = H @ U[..., d]

    return W.transpose(1, 0, 2), W_coefs
