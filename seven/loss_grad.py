"""Gradient and loss functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
import numba

from alphacsc.loss_and_gradient import _compute_DtD_z_i

from .prox import _lazy_prox_positive_L2_ball, _prox_positive_L2_ball_multi


@numba.jit((numba.float64[:, :], numba.float64[:, :], numba.float64[:, :],
            numba.float64[:], numba.float64[:], numba.int64, numba.int64),
            nopython=True, cache=True, fastmath=True)
def _subsampled_cd_iter(C, u, B, step_size, norm_u, offset, block_len):
    """ Main iteration of the subsample scdclinmodel.
    """
    for k in range(u.shape[0]):
        sub_u_k_view = u[k, offset: offset + block_len]
        norm_sub_u_k = sub_u_k_view.dot(sub_u_k_view)
        grad_ = C[k, :].dot(u[:, offset: offset + block_len])
        grad_ -= B[k, offset: offset + block_len]
        sub_u_k_view -= step_size[k] * grad_
        sub_u_k_view, diff_norm_sub_u_k = _lazy_prox_positive_L2_ball(
                                    sub_u_k_view, norm_sub_u_k, norm_u[k])
        norm_u[k] += diff_norm_sub_u_k

# force compilation
_subsampled_cd_iter(np.empty((2, 2)), np.empty((2, 6)), np.empty((2, 6)),
                    np.empty(2), np.empty(2), 0, 2)


def _grad_Lz(Lz, DtD, DtX=None):
    """ Gradient for the temporal prox for multiple voxels.
    """
    grad = _compute_DtD_z_i(z_i=Lz, DtD=DtD)
    if DtX is not None:
        grad -= DtX
    return grad


def obj_u_z(X, lbda, u, Lz, v=None, A=None, valid=True):
    """ Main objective function.
    """
    valid_u = u
    if valid:
        valid_u = _prox_positive_L2_ball_multi(u)

    if A is None:
        if v is None:
            raise ValueError("if A is None, v should be given.")
        A = np.r_[[np.convolve(v, Lz_k) for Lz_k in Lz]].T

    res = (X.T - A.dot(valid_u)).ravel()
    cost = 0.5 * res.dot(res) + lbda * np.sum(np.abs(np.diff(Lz, axis=-1)))

    return cost