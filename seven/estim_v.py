"""Hemodynamic Responses Function models"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from .atlas import get_indices_from_roi
from .utils import lipschitz_est
from .loss_grad import _grad_v_hrf_d_basis
from .optim import proximal_descent
from .convolution import make_toeplitz
from .constants import _precompute_d_basis_constant
from .hrf_model import spm_scaled_hrf


MIN_DELTA = 0.5
MAX_DELTA = 2.0


def _estim_v_scaled_hrf(a, X, z, u, rois_idx, t_r, n_times_atom):
    """ Estimation of v based on the time dilatation SPM HRF. """
    n_hrf_rois, _ = rois_idx.shape
    _, n_times_valid = z.shape
    v = np.empty((n_hrf_rois, n_times_atom))

    for m in range(n_hrf_rois):
        indices = get_indices_from_roi(m, rois_idx)
        uz = u[:, indices].T.dot(z)
        a0 = a[m]

        def h(delta):
            return spm_scaled_hrf(delta=delta, t_r=t_r,
                                    n_times_atom=n_times_atom)

        def cost(delta):
            H = make_toeplitz(h(delta), n_times_valid)
            res = (X[indices, :] - uz.dot(H.T)).ravel()
            return 0.5 * res.dot(res)

        bounds = [(MIN_DELTA + 1.0e-1, MAX_DELTA - 1.0e-1)]

        _a_hat, _, _ = fmin_l_bfgs_b(func=cost, x0=a0, bounds=bounds,
                                    approx_grad=True, maxiter=1000,
                                    pgtol=1.0e-10)

        a[m] = _a_hat
        v[m, :] = h(_a_hat)

    return a, v


def _estim_v_d_basis(a, X, h, z, u, rois_idx):
    """ Estimation of v based on the SPM 3 basis model. """
    n_atoms_hrf, n_times_atom = h.shape
    n_hrf_rois, _ = rois_idx.shape
    _, n_times = X.shape
    _, n_times_valid = z.shape
    v = np.empty((n_hrf_rois, n_times_atom))

    H = np.empty((n_atoms_hrf, n_times, n_times_valid))
    for d in range(n_atoms_hrf):
        H[d, :, :] = make_toeplitz(h[d, :], n_times_valid)

    for m in range(n_hrf_rois):
        indices = get_indices_from_roi(m, rois_idx)
        uz = u[:, indices].T.dot(z)
        n_voxels_rois, _ = uz.shape
        AtA, AtX = _precompute_d_basis_constant(X[indices, :], uz, H)
        a0 = a[m, :]

        def grad(a):
            return _grad_v_hrf_d_basis(a, AtA, AtX)

        n_atoms_hrf = len(a0)
        bounds = np.array(((1, 1), ) + ((0.0, 1.0), ) * (n_atoms_hrf - 1))

        def prox(a, step_size):
            return np.clip(a, bounds[:, 0], bounds[:, 1])

        def AtA_(a):
            return _grad_v_hrf_d_basis(a, AtA)
        step_size = 0.9 / lipschitz_est(AtA_, a0.shape)

        params = dict(x0=a0, grad=grad, prox=prox, step_size=step_size,
                        momentum='fista', restarting=None, max_iter=1000)
        a_m_hat = proximal_descent(**params)

        a[m, :] = a_m_hat
        v[m, :] = a_m_hat.dot(h)

    return a, v
