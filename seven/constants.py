""" Usefull constants computation functions. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
import numba

from .atlas import get_indices_from_roi


@numba.jit((numba.float64[:, :], numba.float64[:, :], numba.int64[:, :]),
            nopython=True, cache=True, fastmath=True)
def _precompute_uvtuv(u, v, rois_idx):
    """ Pre-compute uvtuv.

    Parameters
    ----------
    u : array, shape (n_atoms, n_voxels)
    v : array, shape (n_hrf_rois, n_times_atom)
    rois_idx: array, shape (n_hrf_rois, max indices per rois)

    Return
    ------
    uvtuv : array, shape (n_atoms, n_atoms, 2 * n_times_atom - 1)
    """
    n_atoms, n_voxels = u.shape
    _, n_times_atom = v.shape
    utu = u.dot(u.T)
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
    """ Compute list of B, C from givem H (and X, z). """
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
def _precompute_d_basis_constant(X, uz, H):
    """ Precompute AtA and AtX. """
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