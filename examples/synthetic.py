from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from splinetlsm import SplineDynamicLSM
from splinetlsm.datasets import synthetic_network_mixture
from splinetlsm.mcmc import dynamic_adjacency_to_vec

n_nodes = 500
n_time_points = 10

Y, time_points, X, probas, _ = synthetic_network_mixture(
    n_nodes=n_nodes, n_time_points=n_time_points,
    ls_type='bspline', include_covariates=True,
    tau=0.25, sigma=0.25, intercept=-3, random_state=42)

y_true = dynamic_adjacency_to_vec(Y)
print("Density: {:.3f}".format(y_true.mean()))

model = SplineDynamicLSM(n_features=2, n_knots=20, random_state=42)
model.fit(Y, time_points, X, n_time_points=1)

print("Best AUC: {:.3f}".format(roc_auc_score(y_true.ravel(), probas.ravel())))
print("Model AUC: {:.3f}".format(model.auc_))
print("PPC: {:.3f}".format(pearsonr(
    probas.ravel(), model.probas_.ravel())[0]))
