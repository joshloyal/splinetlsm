import jax.numpy as jnp
import jax
import pandas as pd

import splinetlsm.static_gof as sgof

from numpyro.distributions.util import vec_to_tril_matrix


def tril_vec_to_matrix(x):
    A = vec_to_tril_matrix(x.astype(float), diagonal=-1)
    return A + A.T


def degree(y_vec):
    # y_vec is n_time_steps x n_dyads
    n_time_steps = y_vec.shape[0]

    def stat_fun(carry, t):
        return None, sgof.degree(y_vec[t])
    _, res = jax.lax.scan(stat_fun, None, jnp.arange(n_time_steps))

    return res.astype(int)


def avg_degree(y_vec):
    return degree(y_vec).mean(axis=0)


def density(y_vec):
    return y_vec.mean(axis=1)


def transitivity(y_vec):
    # y_vec is n_time_steps x n_dyads
    n_time_steps = y_vec.shape[0]

    def stat_fun(carry, t):
        return None, sgof.transitivity(y_vec[t])
    _, res = jax.lax.scan(stat_fun, None, jnp.arange(n_time_steps))

    return res


def dyadic_stability(y_vec):
    # y_vec is n_time_steps x n_dyads
    n_time_steps = y_vec.shape[0]

    def stat_fun(carry, t):
        out = (y_vec[t] * y_vec[t+1]).mean()
        out += ((1 - y_vec[t]) * (1 - y_vec[t+1])).mean()
        return None, out
    _, res = jax.lax.scan(stat_fun, None, jnp.arange(n_time_steps-1))

    return res


def edge_persistence(y_vec):
    # y_vec is n_time_steps x n_dyads
    n_time_steps = y_vec.shape[0]

    def stat_fun(carry, t):
        out = (y_vec[t] * y_vec[t+1]).sum() / y_vec[t].sum()
        return None, out
    _, res = jax.lax.scan(stat_fun, None, jnp.arange(n_time_steps-1))

    return res


def nonedge_persistence(y_vec):
    n_time_steps = y_vec.shape[0]

    def stat_fun(carry, t):
        out = ((1 - y_vec[t]) * (1 - y_vec[t+1])).sum() / (1 - y_vec[t]).sum()
        return None, out
    _, res = jax.lax.scan(stat_fun, None, jnp.arange(n_time_steps-1))

    return res


def stat_distribution(stats):
    stat = pd.melt(pd.DataFrame(stats), var_name='t')
    stat['t'] = stat['t'] + 1
    return stat
