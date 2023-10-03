import numpy as np

from scipy.stats import pearsonr
from scipy.special import logit
from sklearn.metrics import roc_auc_score

from splinetlsm import SplineDynamicLSM
from splinetlsm.datasets import synthetic_network_mixture
from splinetlsm.mcmc import dynamic_adjacency_to_vec

n_nodes = 150
n_time_points = 100 
intercept = -4

Y, time_points, X, probas, U, coefs = synthetic_network_mixture(
    n_nodes=n_nodes, n_time_points=n_time_points,
    ls_type='gp', include_covariates=True,
    tau=0.25, sigma=0.25, intercept=intercept, random_state=49)

y_true = dynamic_adjacency_to_vec(Y)
print("Density: {:.3f}".format(y_true.mean()))
#
model = SplineDynamicLSM(n_features=10, n_segments=20, random_state=4, init_type='usvt')
model.fit(Y, time_points, X, n_time_points=0.25)
print(model.n_iter_)
#
print("Best AUC: {:.3f}".format(roc_auc_score(y_true.ravel(), probas.ravel())))
print("Model AUC: {:.3f}".format(model.auc_))
print("PPC: {:.3f}".format(pearsonr(
    probas.ravel(), model.probas_.ravel())[0]))

subdiag = np.tril_indices(n_nodes, k=-1)
error = 0.
for t in range(n_time_points):
    UUt_true = (U[t] @ U[t].T)[subdiag]
    UUt_pred = (model.U_[t] @ model.U_[t].T)[subdiag]
    error += np.mean((UUt_true - UUt_pred) ** 2)
error /= n_time_points
print(np.sqrt(error))

coefs_error = np.sum((model.coefs_ - coefs) ** 2, axis=1)
coefs_error += (model.intercept_ - intercept) ** 2
print(np.sqrt(coefs_error.mean()))

theta_rmse = np.sqrt(np.mean((logit(probas.ravel()) - logit(model.probas_.ravel())) ** 2))
print(theta_rmse)
