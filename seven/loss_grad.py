"""Gradient and loss functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
import numba

from .utils import _compute_uvtuv_z
from .prox import _prox_positive_L2_ball_multi
from .atlas import get_indices_from_roi


def _grad_v_hrf_d_basis(a, AtA, AtX=None):
    """ v d-basis HRF gradient. """
    grad = a.dot(AtA)
    if AtX is not None:
        grad -= AtX
    return grad


def _grad_z(z, uvtuv, uvtX=None):
    """ z gradient. """
    grad = _compute_uvtuv_z(z=z, uvtuv=uvtuv)
    if uvtX is not None:
        grad -= uvtX
    return grad


def _grad_u_k(u, B, C, k, rois_idx):
    """ u gradient. """
    _, n_voxels = B[0, :, :].shape
    grad = np.empty(n_voxels)
    for m in range(rois_idx.shape[0]):
        indices = get_indices_from_roi(m, rois_idx)
        grad[indices] = C[m, k, :].dot(u[:, indices])
        grad[indices] -= B[m, k, indices]
    return grad


@numba.jit((numba.float64[:, :], numba.float64[:, :], numba.float64[:, :],
            numba.int64[:, :]), nopython=True, cache=True, fastmath=True)
def construct_X_hat_from_v(v, z, u, rois_idx):
    """Return X_hat from v, z, u. """
    n_voxels, n_times = u.shape[1], z.shape[1] + v.shape[1] - 1
    uz = z.T.dot(u).T
    X_hat = np.empty((n_voxels, n_times))
    for m in range(rois_idx.shape[0]):
        indices = get_indices_from_roi(m, rois_idx)
        for j in indices:
            X_hat[j, :] = np.convolve(v[m, :], uz[j, :])
    return X_hat


@numba.jit((numba.float64[:, :, :], numba.float64[:, :], numba.float64[:, :],
            numba.int64[:, :]), nopython=True, cache=True, fastmath=True)
def construct_X_hat_from_H(H, z, u, rois_idx):
    """Return X_hat from H, z, u. """
    n_voxels, n_times = u.shape[1], H.shape[1]
    zu = z.T.dot(u)
    X_hat = np.empty((n_voxels, n_times))
    for m in range(rois_idx.shape[0]):
        indices = get_indices_from_roi(m, rois_idx)
        X_hat[indices, :] = H[m, :, :].dot(zu[:, indices]).T
    return X_hat


def _obj(X, u, z, rois_idx, H=None, v=None, valid=True, return_reg=True,
         lbda=None):
    """ Main objective function. """
    u = _prox_positive_L2_ball_multi(u) if valid else u

    if v is not None:
        X_hat = construct_X_hat_from_v(v, z, u, rois_idx)
    if v is None and H is not None:
        X_hat = construct_X_hat_from_H(H, z, u, rois_idx)
    elif H is None and v is None:
        raise ValueError("_obj must have either H or v.")

    residual = (X - X_hat).ravel()
    cost = 0.5 * residual.dot(residual)

    if return_reg:
        if lbda is not None:
            regu = lbda * np.sum(np.abs(np.diff(z, axis=-1)))
            return cost + regu
        else:
            raise ValueError("obj must have lbda to return regularization "
                             "value")
    else:
        return cost
