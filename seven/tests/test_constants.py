"""Testing module for gradient and loss function"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import pytest
import numpy as np

from seven.constants import (_precompute_uvtuv, _precompute_d_basis_constant,
                             _precompute_B_C)
from seven.atlas import get_indices_from_roi
from seven.hrf_model import hrf_3_basis
from seven.convolution import make_toeplitz
from seven.utils import _set_up_test


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_precomputed_uvtuv(seed):
    """ Test the computation of uvtX. """
    kwargs = _set_up_test(seed)
    u, v, rois_idx = kwargs['u'], kwargs['v'], kwargs['rois_idx']

    def _precompute_uvtuv_ref(u, v, rois_idx):
        # computation close to the maths formulation
        n_atoms, n_voxels = u.shape
        _, n_times_atom = v.shape
        uv = np.empty((n_atoms, n_voxels, n_times_atom))
        for m in range(rois_idx.shape[0]):
            indices = get_indices_from_roi(m, rois_idx)
            for j in indices:
                for k in range(n_atoms):
                    uv[k, j, :] = u[k, j] * v[m]
        uvtuv = np.zeros((n_atoms, n_atoms, 2 * n_times_atom - 1))
        for k0 in range(n_atoms):
            for k in range(n_atoms):
                _sum = np.convolve(uv[k0, 0, ::-1], uv[k, 0, :])
                for j in range(1, n_voxels):
                    _sum += np.convolve(uv[k0, j, ::-1], uv[k, j, :])
                uvtuv[k0, k, :] = _sum
        return uvtuv

    uvtuv_ref = _precompute_uvtuv_ref(u, v, rois_idx)
    uvtuv = _precompute_uvtuv(u, v, rois_idx)

    np.testing.assert_allclose(uvtuv_ref, uvtuv)


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_precompute_B_C(seed):
    """ Test the computation of B and C for update u. """
    kwargs = _set_up_test(seed)
    X, z, H, v = kwargs['X'], kwargs['z'], kwargs['H'], kwargs['v']
    rois_idx = kwargs['rois_idx']
    n_hrf_rois, _ = rois_idx.shape

    def _ref_precompute_B_C(X, z, v, rois_idx):
        """ Compute list of B, C from givem H (and X, z). """
        n_hrf_rois, _ = rois_idx.shape
        n_voxels, _ = X.shape
        n_atoms, n_times_valid = z.shape
        vtvz = np.empty((n_atoms, n_times_valid))
        B = np.empty((n_hrf_rois, n_atoms, n_voxels))
        C = np.empty((n_hrf_rois, n_atoms, n_atoms))
        for m in range(n_hrf_rois):
            indices = get_indices_from_roi(m, rois_idx)
            for k in range(n_atoms):
                vz_k = np.convolve(v[m, :], z[k, :])
                vtvz[k, :] = np.convolve(v[m, ::-1], vz_k, mode='valid')
            zvtvz = vtvz.dot(z.T)
            C[m, :, :] = zvtvz
            for j in indices:
                vtX = np.convolve(v[m, ::-1], X[j, :], mode='valid')
                B[m, :, j] = z.dot(vtX.T)
        return B, C

    B, C = _precompute_B_C(X, z, H, rois_idx)
    ref_B, ref_C = _ref_precompute_B_C(X, z, v, rois_idx)

    for m in range(n_hrf_rois):
        indices = get_indices_from_roi(m, rois_idx)
        np.testing.assert_allclose(ref_B[m, :, indices], B[m, :, indices])
    np.testing.assert_allclose(ref_C, C)


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_precompute_d_basis_constant(seed):
    """ Test the computation of AtA and AtX for the 3 HRF basis. """
    kwargs = _set_up_test(seed)
    t_r, n_times_atom = kwargs['t_r'], kwargs['n_times_atom']
    X, u, z = kwargs['X'], kwargs['u'], kwargs['z']
    uz = u.T.dot(z)
    h = hrf_3_basis(t_r, n_times_atom)
    _, n_times_valid = z.shape
    n_voxels_rois, n_times = X.shape
    n_atoms_hrf, _ = h.shape
    A_d = np.empty((n_voxels_rois, n_times, n_atoms_hrf))
    for d in range(n_atoms_hrf):
        for j in range(n_voxels_rois):
            A_d[j, :, d] = np.convolve(h[d, :], uz[j, :])
    AtX = np.array([A_d[:, :, d].ravel().dot(X.ravel())
                    for d in range(n_atoms_hrf)])
    AtA = np.empty((n_atoms_hrf, n_atoms_hrf))
    for d0 in range(n_atoms_hrf):
        for d1 in range(n_atoms_hrf):
            AtA[d0, d1] = A_d[:, :, d0].ravel().dot(A_d[:, :, d1].ravel())

    H = np.empty((n_atoms_hrf, n_times, n_times_valid))
    for d in range(n_atoms_hrf):
        H[d, :, :] = make_toeplitz(h[d, :], n_times_valid)
    AtA_, AtX_ = _precompute_d_basis_constant(X, uz, H)

    np.testing.assert_allclose(AtA, AtA_)
    np.testing.assert_allclose(AtX, AtX_)
