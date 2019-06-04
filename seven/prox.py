"""Proximity operator functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
import numba
from prox_tv import tv1_1d


@numba.jit((numba.float64[:], numba.float64, numba.float64), nopython=True,
           cache=True, fastmath=True)
def _lazy_prox_positive_L2_ball(sub_u_k, norm_sub_u_k, norm_u_k):
    """_lazy_prox_positive_L2_ball

    Low computation of prox-op for: I{ u_kj > 0 and ||u_k||_2^2 =< 1.0}.

    Return:
        - prox(u[offset: offset + length])
        - the norm difference of u[offset: offset + length] after the proximal
        operator.
    """
    available_norm = norm_sub_u_k + (1.0 - norm_u_k)
    block_len = sub_u_k.shape[0]
    norm_ = 0.0
    for j in range(block_len):
        if sub_u_k[j] < 0.0:
            sub_u_k[j] = 0.0
        else:
            norm_ += sub_u_k[j] * sub_u_k[j]
    if norm_ < np.finfo(np.float64).eps:
        return sub_u_k, np.finfo(np.float64).eps - norm_sub_u_k
    ratio = np.sqrt(available_norm / norm_)
    if ratio <= 1.0:
        for j in range(block_len):
            sub_u_k[j] *= ratio
        return sub_u_k, available_norm - norm_sub_u_k
    else:
        return sub_u_k, norm_ - norm_sub_u_k


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


# force Numba compilation
_lazy_prox_positive_L2_ball(np.random.randn(2), 0.9, 0.5)
_prox_positive_L2_ball(np.random.randn(5))


def _prox_positive_L2_ball_multi(u):
    """ """
    return np.r_[[_prox_positive_L2_ball(u_k) for u_k in u]]


def _prox_tv(Lz, lbda, step_size):
    """ Projection of the infinity norm ball. """
    return tv1_1d(Lz, lbda * step_size)


def _prox_tv_multi(Lz, lbda, step_size):
    """ """
    return np.vstack([_prox_tv(Lz_k, lbda, step_size) for Lz_k in Lz])
