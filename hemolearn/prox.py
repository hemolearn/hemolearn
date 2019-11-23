"""Proximity module: the proximal operator functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
import numba
from prox_tv import tv1_1d


@numba.jit((numba.float64[:], numba.float64,), nopython=True, cache=True,
           fastmath=True)
def _prox_l1_simplex(u_i, eta):  # pragma: no cover
    """ _prox_l1_simplex
    prox-op for: I{ u_ij > 0 and sum_j u_ij = eta}(u_i)

    Parameters
    ----------
    u : array, shape (n_voxels,), the spatial map
    step_size : float, the step-size for the gradient descent

    Return
    ------
    prox_u : array, shape (n_voxels,), the valid approximated
        the spatial map
    """
    s = np.sort(u_i)[::-1]
    c = (np.cumsum(s) - eta) / np.arange(1, len(u_i)+1)
    if len([s > c]) > 0:
        m = np.arange(len(u_i))[s > c].max()
        return u_i - np.minimum(u_i, c[m])
    else:
        p_u_i = np.zeros_like(u_i)
        p_u_i[np.argmax(u_i)] = np.max(u_i)
        return p_u_i


@numba.jit((numba.float64[:], numba.float64,), nopython=True, cache=True,
           fastmath=True)
def _prox_positive(u_k, step_size):  # pragma: no cover
    """_prox_positive,
    Full computation of prox-op for: I{ u_kj > 0 }.

    Parameters
    ----------
    u_k : array, shape (n_voxels, ), one spatial map

    Return
    ------
    prox_u_k : array, shape (n_voxels, ), one valid spatial map
    """
    p = u_k.shape[0]
    for j in range(p):
        if u_k[j] < 0.0:
            u_k[j] = 0.0
    return u_k


@numba.jit((numba.float64[:], numba.float64,), nopython=True, cache=True,
           fastmath=True)
def _prox_positive_l2_ball(u_k, step_size):  # pragma: no cover
    """_prox_positive_L2_ball,
    Full computation of prox-op for: I{ u_kj > 0 and ||u_k||_2^2 =< 1.0}.

    Parameters
    ----------
    u_k : array, shape (n_voxels, ), one spatial map

    Return
    ------
    prox_u_k : array, shape (n_voxels, ), one valid spatial map
    """
    p = u_k.shape[0]
    norm_u_k = 0.0
    for j in range(p):
        if u_k[j] < 0.0:
            u_k[j] = 0.0
        else:
            norm_u_k += u_k[j] * u_k[j]
    if norm_u_k > 1.0:
        u_k /= norm_u_k
    return u_k


def _prox_tv_multi(z, lbda, step_size):
    """ _prox_tv_multi

    Parameters
    ----------
    z : array, shape (n_atoms, n_times_valid), temporal components
    lbda : float, the temporal regularization parameter
    step_size : float, the step-size for the gradient descent

    Return
    ------
    prox_z : array, shape (n_atoms, n_times_valid), the valid approximated
        temporal components
    """
    return np.vstack([tv1_1d(z_k, lbda * step_size) for z_k in z])
