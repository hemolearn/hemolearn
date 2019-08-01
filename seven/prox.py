"""Proximity operator functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
import numba
from prox_tv import tv1_1d


@numba.jit((numba.float64[:],), nopython=True, cache=True, fastmath=True)
def _prox_positive_L2_ball(u_k):
    """_prox_positive_L2_ball

    Full computation of prox-op for: I{ u_kj > 0 and ||u_k||_2^2 =< 1.0}.

    Return:
        - prox(u)
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


_prox_positive_L2_ball(np.random.randn(5))


def _prox_positive_L2_ball_multi(u):
    """ _prox_positive_L2_ball_multi. """
    return np.r_[[_prox_positive_L2_ball(u_k) for u_k in u]]


def _prox_tv_multi(Lz, lbda, step_size):
    """ _prox_tv_multi. """
    return np.vstack([tv1_1d(Lz_k, lbda * step_size) for Lz_k in Lz])
