"""Testing module for the utility functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import pytest
import numpy as np
from hemolearn.checks import check_random_state
from hemolearn.estim_v import _estim_v_scaled_hrf, _estim_v_d_basis
from hemolearn.hrf_model import scaled_hrf, hrf_3_basis, MIN_DELTA, MAX_DELTA
from hemolearn.atlas import split_atlas
from hemolearn.loss_grad import construct_X_hat_from_v


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_estim_v_scaled_hrf(seed):
    """ Test the estimation of the HRF with the scaled HRF model without noise
    in the observed data. """
    rng = check_random_state(seed)
    eps = 1.0e-3
    t_r = 1.0
    n_atoms = 2
    n_times_valid = 500
    n_times_atom = 60
    n_voxels_in_rois = 200
    n_hrf_rois = 5
    n_voxels = n_voxels_in_rois * n_hrf_rois
    indices = np.arange(n_voxels)
    rois_s = np.split(indices, n_hrf_rois)
    hrf_rois = dict(zip(range(1, n_hrf_rois + 1), rois_s))
    u = rng.randn(n_atoms, n_voxels)
    z = rng.randn(n_atoms, n_times_valid)
    rois_idx, _, _ = split_atlas(hrf_rois)
    a_true = rng.uniform(MIN_DELTA + eps, MAX_DELTA - eps, n_hrf_rois)
    v_true = np.c_[[scaled_hrf(a_, t_r, n_times_atom) for a_ in a_true]]
    X = construct_X_hat_from_v(v_true, z, u, rois_idx)

    a_init = rng.uniform(MIN_DELTA + eps, MAX_DELTA - eps, n_hrf_rois)
    a_hat, v_hat = _estim_v_scaled_hrf(a_init, X, z, u, rois_idx, t_r,
                                       n_times_atom)

    # no garantie of recovery in any case...
    np.testing.assert_allclose(a_true, a_hat, atol=1e-1)
    np.testing.assert_allclose(v_true, v_hat, atol=1e-1)


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_estim_v_3_basis(seed):
    """ Test the estimation of the HRF with the scaled HRF model without noise
    in the observed data. """
    rng = check_random_state(seed)
    t_r = 1.0
    n_atoms = 2
    n_voxels = 1000
    n_times_valid = 100
    n_times_atom = 30
    indices = np.arange(n_voxels)
    n_hrf_rois = 2
    rois_1 = indices[int(n_voxels/2):]
    rois_2 = indices[:int(n_voxels/2)]
    hrf_rois = {1: rois_1, 2: rois_2}

    u = rng.randn(n_atoms, n_voxels)
    z = rng.randn(n_atoms, n_times_valid)
    rois_idx, _, _ = split_atlas(hrf_rois)
    h = hrf_3_basis(t_r, n_times_atom)
    a_true = np.c_[[[1.0, 0.8, 0.5], [1.0, 0.5, 0.0]]]
    v_true = np.c_[[a_.dot(h) for a_ in a_true]]
    X = construct_X_hat_from_v(v_true, z, u, rois_idx)

    a_init = np.c_[[np.array([1.0, 0.0, 0.0]) for _ in range(n_hrf_rois)]]
    a_hat, v_hat = _estim_v_d_basis(a_init, X, h, z, u, rois_idx)

    # no garantie of recovery in any case...
    np.testing.assert_allclose(a_true, a_hat, atol=1e-1)
    np.testing.assert_allclose(v_true, v_hat, atol=1e-1)
