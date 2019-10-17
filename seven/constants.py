""" Constants module: Usefull constants computation functions for gradient
descent algorithm. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
import numba

from .atlas import get_indices_from_roi


def _precompute_sum_ztz_sum_ztz_y(uz_roi, X_roi, n_times_atom, factor):
    """ Precompute usefull constants for _grad_v_scaled_hrf

    Parameters
    ----------
    X_roi : array, shape (n_voxels_roi, n_times), fMRI data
    uz_roi : array, shape (n_voxels_roi, n_times_valid), neural activty signals

    Return
    ------
    sum_ztz : array, shape (2 * n_times_valid - 1,), sum of auto-convolution
        for each temporal activation (for each voxels)
    sum_ztz_y : array or None, shape (n_times_atom,), sum of auto-convolution
        for each temporal activation convolved with x_j (for each voxels)
    """
    n_voxels_roi, n_times_valid = uz_roi.shape
    sum_ztz = np.zeros((2 * n_times_atom - 1,))
    sum_ztz_y = np.zeros((n_times_atom,))
    uz_roi_padded = np.zeros(2 * (n_times_atom - 1) + n_times_valid)
    for j in range(n_voxels_roi):
        _indices = slice(n_times_atom - 1, n_times_atom - 1 + n_times_valid)
        uz_roi_padded[_indices] = uz_roi[j, :]
        sum_ztz += np.convolve(uz_roi[j, :][::-1], uz_roi_padded, 'valid')
        # In case we try to recover the cost-function from the gradient,
        # the formula needs a '2 *' before 'the X_roi[j, :]', thus the factor
        # option (equal to 1.0 in case of gradient compuation)
        sum_ztz_y += np.convolve(uz_roi[j, :][::-1], factor * X_roi[j, :],
                                 'valid')
    return sum_ztz, sum_ztz_y


@numba.jit((numba.float64[:, :], numba.float64[:, :], numba.int64[:, :]),
           nopython=True, cache=True, fastmath=True)
def _precompute_uvtuv(u, v, rois_idx):  # pragma: no cover
    """ Pre-compute uvtuv.

    Parameters
    ----------
    u : array, shape (n_atoms, n_voxels), spatial maps
    v : array, shape (n_hrf_rois, n_times_atom), HRFs
    rois_idx: array, shape (n_hrf_rois, max_indices_per_rois), HRF ROIs

    Return
    ------
    uvtuv : array, shape (n_atoms, n_atoms, 2 * n_times_atom - 1), precomputed
        operator
    """
    n_atoms, n_voxels = u.shape
    _, n_times_atom = v.shape
    vtv = np.empty((n_voxels, 2 * n_times_atom - 1))
    for m in range(rois_idx.shape[0]):
        indices = get_indices_from_roi(m, rois_idx)
        vtv[indices, :] = np.convolve(v[m, :], v[m, ::-1])
    uvtuv = np.zeros((n_atoms, n_atoms, 2 * n_times_atom - 1))
    for k0 in range(n_atoms):
        for k in range(n_atoms):
            _sum = u[k0, 0] * u[k, 0] * vtv[0]
            for j in range(1, n_voxels):
                _sum += u[k0, j] * u[k, j] * vtv[j, :]
            uvtuv[k0, k, :] = _sum
    return uvtuv


def _precompute_B_C(X, z, H, rois_idx):
    """ Compute list of B, C from givem H (and X, z).

    Parameters
    ----------
    X : array, shape (n_voxels, n_times), fMRI data
    z : array, shape (n_atoms, n_times_valid), temporal components
    H : array, shape (n_hrf_rois, n_times_valid, n_times), Toeplitz matrices
    rois_idx: array, shape (n_hrf_rois, max_indices_per_rois),

    Return
    ------
    B : array, shape (n_atoms, n_voxels), precomputed operator
    C : array, shape (n_atoms, n_atoms), precomputed operator
    """
    n_hrf_rois, _ = rois_idx.shape
    n_voxels, _ = X.shape
    n_atoms, _ = z.shape
    B = np.empty((n_hrf_rois, n_atoms, n_voxels))
    C = np.empty((n_hrf_rois, n_atoms, n_atoms))
    for m in range(n_hrf_rois):
        indices = get_indices_from_roi(m, rois_idx)
        H_m = H[m, :, :]
        z_H_m_t = z.dot(H_m.T)
        C[m, :, :] = z_H_m_t.dot(z_H_m_t.T)
        B[m, :, indices] = X[indices, :].dot(z_H_m_t.T)
    return B, C


@numba.jit((numba.float64[:, :], numba.float64[:, :], numba.float64[:, :, :]),
           nopython=True, cache=True, fastmath=True)
def _precompute_d_basis_constant(X, uz, H):  # pragma: no cover
    """ Precompute AtA and AtX.

    Parameters
    ----------
    X : array, shape (n_voxels, n_times), fMRI data
    uz : array, shape (n_voxels, n_times_valid), neural activity
    H : array, shape (n_hrf_rois, n_times_valid, n_times), Toeplitz matrices

    Return
    ------
    AtA : array, shape (n_atoms_hrf, n_atoms_hrf), precomputed operator
    AtX : array, shape (n_atoms_hrf), precomputed operator
    """
    n_voxels_rois, n_times = X.shape
    n_atoms_hrf, _, _ = H.shape
    A_d = np.empty((n_atoms_hrf, n_voxels_rois * n_times))
    AtX = np.empty(n_atoms_hrf)
    AtA = np.empty((n_atoms_hrf, n_atoms_hrf))
    for d in range(n_atoms_hrf):
        A_d[d, :] = uz.dot(H[d, :, :].T).ravel()
    X_ravel = X.ravel()
    for d in range(n_atoms_hrf):
        AtX[d] = A_d[d, :].dot(X_ravel)
        for d1 in range(d, n_atoms_hrf):
            AtA[d, d1] = AtA[d1, d] = A_d[d, :].dot(A_d[d1, :])
    return AtA, AtX
