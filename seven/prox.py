"""Proximity module: the proximal operator functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
import numba
from prox_tv import tv1_1d


@numba.jit((numba.float64[:],), nopython=True, cache=True, fastmath=True)
def _prox_positive_L2_ball(u_k):  # pragma: no cover
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


def _prox_positive_L2_ball_multi(u):
    """ _prox_positive_L2_ball_multi,
    Full computation of prox-op for: I{ u_kj > 0 and ||u_k||_2^2 =< 1.0} for
    each spatial map u_k.

    Parameters
    ----------
    u : array, shape (n_atoms, n_voxels), spatial maps

    Return
    ------
    prox_u_k : array, shape (n_atoms, n_voxels), the valid spatial maps
    """
    return np.r_[[_prox_positive_L2_ball(u_k) for u_k in u]]


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
