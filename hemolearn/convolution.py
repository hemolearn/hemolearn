""" Convolution module: gathers functions that define a convolutional operator.
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
import numba
from scipy import linalg
from .atlas import get_indices_from_roi


@numba.jit((numba.float64[:, :], numba.float64[:, :], numba.float64[:, :],
            numba.int64[:, :]), nopython=True, cache=True, fastmath=True)
def adjconv_uv(residual_i, u, v, rois_idx):  # pragma: no cover
    """ Pre-compute the convolution residual_i with the transpose for each
    atom k.

    Parameters
    ----------
    residual_i : array, shape (n_voxels, n_times) residual term in the gradient
    u : array, shape (n_atoms, n_voxels) spatial maps
    v : array, shape (n_hrf_rois, n_times_atom) HRFs
    rois_idx: array, shape (n_hrf_rois, max_indices_per_rois), HRF ROIs

    Return
    ------
    uvtX : array, shape (n_atoms, n_times_valid) computed operator image
    """
    _, n_times_atom = v.shape
    n_voxels, n_time = residual_i.shape
    vtX = np.empty((n_voxels, n_time - n_times_atom + 1))
    for m in range(rois_idx.shape[0]):
        for j in get_indices_from_roi(m, rois_idx):
            vtX[j, :] = np.correlate(residual_i[j, :], v[m, :])
    return np.dot(u, vtX)


def adjconv_uH(residual, u, H, rois_idx):
    """ Pre-compute the convolution residual with the transpose for each
    atom k.

    Parameters
    ----------
    residual : array, shape (n_voxels, n_times) residual term in the gradient
    u : array, shape (n_atoms, n_voxels) spatial maps
    H : array, shape (n_hrf_rois, n_times_valid, n_times), Toeplitz matrices
    rois_idx: array, shape (n_hrf_rois, max_indices_per_rois), HRF ROIs

    Return
    ------
    uvtX : array, shape (n_atoms, n_times_valid) computed operator image
    """
    n_hrf_rois, _, n_times_valid = H.shape
    n_voxels, n_time = residual.shape
    vtX = np.empty((n_voxels, n_times_valid))
    for m in range(n_hrf_rois):
        indices = get_indices_from_roi(m, rois_idx)
        vtX[indices, :] = residual[indices, :].dot(H[m, :, :])
    return np.dot(u, vtX)


def make_toeplitz(v, n_times_valid):
    """ Make Toeplitz matrix from given kernel to perform valid
    convolution.

    Parameters
    ----------
    v : array, shape (n_times_atom), HRF
    n_times_valid : int, length of the temporal components

    Return
    ------
    H : array, shape (n_times, n_times_valid), Toeplitz matrix, recall that
        n_times = n_times_valid + n_times_atom -1
    """
    padd = np.zeros((1, n_times_valid - 1))
    return linalg.toeplitz(np.c_[v[None, :], padd], np.c_[1.0, padd])
