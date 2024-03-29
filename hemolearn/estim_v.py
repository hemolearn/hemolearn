"""Hemodynamic Responses Function models"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
from scipy import optimize

from .atlas import get_indices_from_roi
from .loss_grad import _grad_v_hrf_d_basis, _grad_v_scaled_hrf, _loss_v
from .convolution import make_toeplitz
from .constants import (_precompute_d_basis_constant,
                        _precompute_sum_ztz_sum_ztz_y)
from .hrf_model import scaled_hrf, MIN_DELTA, MAX_DELTA


def _estim_v_scaled_hrf(a, X, z, u, rois_idx, t_r, n_times_atom):
    """ Estimation of v based on the time dilatation SPM HRF.

    Parameters
    ----------
    a : array, shape (n_hrf_rois, n_param_HRF), initial HRF parameters
    X : array, shape (n_voxels, n_times), fMRI data
    z : array, shape (n_atoms, n_times_valid), temporal components
    u : array, shape (n_atoms, n_voxels), spatial maps
    rois_idx: array, shape (n_hrf_rois, max_indices_per_rois), HRF ROIs
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    n_times_atom : int, (default=30), number of points on which represent the
        Haemodynamic Response Function (HRF), this leads to the duration of the
        response function, duration = n_times_atom * t_r

    Return
    ------
    a : array, shape (n_hrf_rois, n_param_HRF), estimated HRF parameters
    v : array, shape (n_hrf_rois, n_times_atom), estimated HRFs
    """
    n_hrf_rois, _ = rois_idx.shape
    v = np.empty((n_hrf_rois, n_times_atom))

    for m in range(n_hrf_rois):
        indices = get_indices_from_roi(m, rois_idx)
        X_roi = X[indices, :]
        u_roi = u[:, indices]
        uz_roi = u_roi.T.dot(z)
        a0 = np.array([a[m], ])

        sum_ztz, sum_ztz_y = _precompute_sum_ztz_sum_ztz_y(uz_roi, X_roi,
                                                           n_times_atom,
                                                           factor=1.0)
        sum_ztz_, sum_ztz_y_ = _precompute_sum_ztz_sum_ztz_y(uz_roi, X_roi,
                                                             n_times_atom,
                                                             factor=2.0)

        def grad(a_):
            _grad = _grad_v_scaled_hrf(a_, t_r, n_times_atom, sum_ztz,
                                       sum_ztz_y)
            return np.array([_grad, ])

        def f(a_):
            cost = _loss_v(a_, u_roi, z, X_roi, t_r, n_times_atom, sum_ztz_,
                           sum_ztz_y_)
            return cost

        _a_hat, _, _ = optimize.fmin_l_bfgs_b(
                                        func=f, x0=a0, fprime=grad,
                                        bounds=[(MIN_DELTA, MAX_DELTA), ])

        a[m] = _a_hat
        v[m, :] = scaled_hrf(_a_hat, t_r, n_times_atom)

    return a, v


def _estim_v_d_basis(a, X, h, z, u, rois_idx):
    """ Estimation of v based on the SPM 3 basis model.

    Parameters
    ----------
    a : array, shape (n_hrf_rois, n_param_HRF), initial HRF parameters
    X : array, shape (n_voxels, n_times), fMRI data
    h : array, shape (n_atoms_hrf, n_times_atom), HRF basis
    z : array, shape (n_atoms, n_times_valid), temporal components
    u : array, shape (n_atoms, n_voxels), spatial maps
    rois_idx: array, shape (n_hrf_rois, max_indices_per_rois), HRF ROIs

    Return
    ------
    a : array, shape (n_hrf_rois, n_param_HRF), estimated HRF parameters
    v : array, shape (n_hrf_rois, n_times_atom), estimated HRFs
    """
    n_atoms_hrf, n_times_atom = h.shape
    n_hrf_rois, _ = rois_idx.shape
    _, n_times = X.shape
    n_atoms, n_times_valid = z.shape
    v = np.empty((n_hrf_rois, n_times_atom))

    H = np.empty((n_atoms_hrf, n_times, n_times_valid))
    for d in range(n_atoms_hrf):
        H[d, :, :] = make_toeplitz(h[d, :], n_times_valid)

    for m in range(n_hrf_rois):
        indices = get_indices_from_roi(m, rois_idx)
        u_rois = u[:, indices].T
        X_rois = X[indices, :]
        uz = u_rois.dot(z)
        n_voxels_rois, _ = uz.shape
        AtA, AtX = _precompute_d_basis_constant(X_rois, uz, H)
        a0 = a[m, :]

        def grad(a):
            return _grad_v_hrf_d_basis(a, AtA, AtX)

        def obj(a):
            v = a.dot(h)
            v_z = np.empty((n_atoms, n_times_valid + n_times_atom - 1))
            for k in range(z.shape[0]):
                v_z[k, :] = np.convolve(v, z[k, :])
            X_hat = u_rois.dot(v_z)
            residual = (X_hat - X_rois).ravel()
            return 0.5 * residual.dot(residual)

        n_atoms_hrf = len(a0)
        bounds = np.array(((1, 1), ) + ((0.0, 1.0), ) * (n_atoms_hrf - 1))

        params = dict(func=obj, x0=a0, fprime=grad, bounds=bounds)

        a_m_hat, _, _ = optimize.fmin_l_bfgs_b(**params)

        a[m, :] = a_m_hat
        v[m, :] = a_m_hat.dot(h)

    return a, v
