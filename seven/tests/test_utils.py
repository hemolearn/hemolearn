"""Testing module for the utility functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import pytest
import numpy as np
from seven.checks import check_random_state
from seven.constants import _precompute_uvtuv
from seven.atlas import get_indices_from_roi
from seven.utils import (get_nifti_ext, lipschitz_est, tp, fwhm,
                         _compute_uvtuv_z)
from seven.utils import _set_up_test


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_get_nifti_ext(seed):
    """ Test the extraction of the extension. """
    exts = ['.nii', '.nii', '.nii.gz']
    fnames = ['foo/bar/file.nii', 'file.nii', 'file.nii.gz']
    for fname, ext in zip(fnames, exts):
        _, ext_ = get_nifti_ext(fname)
        assert ext == ext_


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_lipschitz_est(seed):
    """ Test the estimation of the Lipschitz. """
    rng = check_random_state(seed)

    # test the identity case
    def AtA(x):
        return x
    shape = (10,)
    L = lipschitz_est(AtA, shape)
    assert 1.0 == pytest.approx(L)

    # test by the definition of the Lipschitz constant
    A = rng.randn(10, 10)
    AtA_ = A.T.dot(A)
    def AtA(x):
        return AtA_.dot(x)
    L = lipschitz_est(AtA, shape)
    for _ in range(100):
        x, y = rng.randn(10), rng.randn(10)
        a = np.linalg.norm(AtA(x) - AtA(y))
        b = np.linalg.norm(x - y)
        assert a <= L * b


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_tp(seed):
    """ Test the estimation of the Time to Peak. """
    t = np.linspace(-10.0, 10.0, 1.0e5)
    mu = 0.0
    sigma = 1.0
    d = 1.0 / (sigma * np.sqrt(2.0 * np.pi)) *\
                                  np.exp( - (t - mu) ** 2 / (2.0 * sigma ** 2))
    np.testing.assert_allclose(tp(t, d), 0.0, atol=1.0e-3)


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_fwhm(seed):
    """ Test the estimation of the Full Width at Half Max. """
    t = np.linspace(-10.0, 10.0, 1.0e5)
    mu = 0.0
    sigma = 1.0
    d = 1.0 / (sigma * np.sqrt(2.0 * np.pi)) *\
                                  np.exp( - (t - mu) ** 2 / (2.0 * sigma ** 2))
    np.testing.assert_allclose(fwhm(t, d), 2 * np.sqrt(2 * np.log(2)))


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_compute_uvtuv_z(seed):
    """ Test the computation of uvtuv_z. """
    kwargs = _set_up_test(seed)
    z, u, v = kwargs['z'], kwargs['u'], kwargs['v']
    rois_idx, n_voxels = kwargs['rois_idx'], kwargs['n_voxels']

    n_hrf_rois, _ = rois_idx.shape
    _, n_times_valid = z.shape

    uz = z.T.dot(u).T
    vtuv_z = np.empty((n_voxels, n_times_valid))
    for m in range(n_hrf_rois):
        indices = get_indices_from_roi(m, rois_idx)
        for j in indices:
            uv_z_j = np.convolve(v[m, :], uz[j, :])
            vtuv_z[j, :] = np.convolve(v[m, ::-1], uv_z_j, mode='valid')
    uvtuv_z_ref = u.dot(vtuv_z)

    uvtuv = _precompute_uvtuv(u, v, rois_idx)
    uvtuv_z = _compute_uvtuv_z(z, uvtuv)

    np.testing.assert_allclose(uvtuv_z_ref, uvtuv_z)
