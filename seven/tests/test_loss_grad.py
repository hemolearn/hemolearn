"""Testing module for gradient and loss function"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import pytest
import numpy as np
from scipy.optimize import approx_fprime

from seven.hrf_model import spm_hrf, spm_hrf_3_basis
from seven.atlas import split_atlas
from seven.constants import _precompute_B_C, _precompute_uvtuv
from seven.checks import check_random_state
from seven.loss_grad import (_grad_u_k, _grad_z, _grad_v_hrf_d_basis,
                             _obj, construct_X_hat_from_v,
                             construct_X_hat_from_H)
from seven.convolution import adjconv_uH, make_toeplitz
from seven.utils import _set_up_test


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_construct_X_hat(seed):
    """ Test the X construction functions. """
    kwargs = _set_up_test(seed)
    n_atoms, n_voxels = kwargs['n_atoms'], kwargs['n_voxels']
    X, u, z = kwargs['X'], kwargs['u'], kwargs['z']
    v, H = kwargs['v'], kwargs['H']
    rois_idx = kwargs['rois_idx']
    X_hat = construct_X_hat_from_v(v, z, u, rois_idx)
    X_hat_ = construct_X_hat_from_H(H, z, u, rois_idx)
    np.testing.assert_allclose(X_hat, X_hat_)


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_loss(seed):
    """ Test the loss function. """
    kwargs = _set_up_test(seed)
    n_atoms, n_voxels = kwargs['n_atoms'], kwargs['n_voxels']
    X, u, z = kwargs['X'], kwargs['u'], kwargs['z']
    v, H = kwargs['v'], kwargs['H']
    rois_idx = kwargs['rois_idx']

    def loss_ref(X, u, z, v, rois_idx):
        res = (X - construct_X_hat_from_v(v, z, u, rois_idx)).ravel()
        return 0.5 * res.dot(res)

    loss_ref_ = loss_ref(X, u, z, v, rois_idx)
    loss_test_ = _obj(X, u, z, rois_idx, H=H, valid=False,
                      return_reg=False, lbda=None)

    np.testing.assert_allclose(loss_ref_, loss_test_)


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_grad_u(seed):
    """ Test the gradient of u. """
    kwargs = _set_up_test(seed)
    n_atoms, n_voxels = kwargs['n_atoms'], kwargs['n_voxels']
    X, H, u, z = kwargs['X'], kwargs['H'], kwargs['u'], kwargs['z']
    rois_idx = kwargs['rois_idx']
    B, C = kwargs['B'], kwargs['C']

    # Finite difference with one HRF
    def finite_grad_one_hrfs_(u):
        def f(u):
            u = u.reshape((n_atoms, n_voxels))
            return _obj(X, u, z, rois_idx, H=H, valid=False,
                        return_reg=False, lbda=None)
        grad_ = approx_fprime(xk=u.ravel(), f=f, epsilon=1.0e-6)
        return grad_.reshape((n_atoms, n_voxels))

    finite_grad_one_hrfs = finite_grad_one_hrfs_(u)

    # Multiple (identical) HRFs
    grad_multi_hrfs = np.empty((n_atoms, n_voxels))
    for k in range(n_atoms):
        grad_multi_hrfs[k, :] = _grad_u_k(u, B, C, k, rois_idx)

    np.testing.assert_allclose(finite_grad_one_hrfs, grad_multi_hrfs,
                               rtol=1e-5, atol=1e-3)


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_grad_z(seed):
    """ Test the gradient of z. """
    kwargs = _set_up_test(seed)
    n_atoms, n_times_valid = kwargs['n_atoms'], kwargs['n_times_valid']
    X, u, z, v,  = kwargs['X'], kwargs['u'], kwargs['z'], kwargs['v']
    H = kwargs['H']
    rois_idx = kwargs['rois_idx']
    uvtuv = _precompute_uvtuv(u=u, v=v, rois_idx=rois_idx)
    uvtX = adjconv_uH(X, u=u, H=H, rois_idx=rois_idx)

    # Finite grad z
    def finite_grad_z(z):
        def f(z):
            z = z.reshape((n_atoms, n_times_valid))
            return _obj(X, u, z, rois_idx, H=H, valid=False,
                        return_reg=False, lbda=None)
        grad_ = approx_fprime(xk=z.ravel(), f=f, epsilon=1.0e-6)
        return grad_.reshape((n_atoms, n_times_valid))

    grad_ref = finite_grad_z(z)

    # Closed form grad z
    grad_ = _grad_z(z, uvtuv=uvtuv, uvtX=uvtX)

    np.testing.assert_allclose(grad_ref, grad_, rtol=1e-5, atol=1e-3)


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_grad_v_hrf_d_basis(seed):
    """ Test the gradient of v (model: hrf 3 basis). """
    rng = check_random_state(seed)
    kwargs = _set_up_test(seed)
    t_r, n_times_atom = kwargs['t_r'], kwargs['n_times_atom']
    X, u, z = kwargs['X'], kwargs['u'], kwargs['z']
    h = spm_hrf_3_basis(t_r, n_times_atom)
    uz = u.T.dot(z)
    n_voxels_rois, n_times_valid = uz.shape
    n_atoms_hrf, _ = h.shape
    _, n_times = X.shape
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
    a = np.array([1.0, 0.5, 0.3])

    # Finite grad v
    def finite_grad_v(a):
        def f(a):
            X_hat = a[0] * A_d[:, :, 0]
            for d in range(1, n_atoms_hrf):
                X_hat += a[d] * A_d[:, :, d]
            res = (X - X_hat).ravel()
            return 0.5 * res.dot(res)
        return approx_fprime(xk=a, f=f, epsilon=1.0e-6)

    grad_ref = finite_grad_v(a)

    # Closed form grad v
    grad_ = _grad_v_hrf_d_basis(a, AtA, AtX)

    np.testing.assert_allclose(grad_ref, grad_, rtol=1e-5, atol=1e-3)
