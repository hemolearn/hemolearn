"""Loss and Gradients module: gradient and cost-functions functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
import numba

from .utils import _compute_uvtuv_z
from .atlas import get_indices_from_roi
from .hrf_model import delta_derivative_double_gamma_hrf, scaled_hrf


def _loss_v(a, u, z, X, t_r, n_times_atom, sum_ztz=None, sum_ztz_y=None):
    """ Loss fonction (simple quadratic data fidelity term) for scaled HRF
    estimation.

    Parameters
    ----------
    a : array, shape (n_hrf_rois, n_param_HRF), init. HRF parameters
    u : array, shape (n_atoms, n_voxels), spatial maps
    z : array, shape (n_atoms, n_times_valid), temporal components
    X : array, shape (n_voxels, n_times), fMRI data
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    n_times_atom : int, (default=30), number of points on which represent the
        Haemodynamic Response Function (HRF), this leads to the duration of the
        response function, duration = n_times_atom * t_r

    Return
    ------
    loss : float, the cost-function evaluated on a
    """
    # sufficent statistic case
    if (sum_ztz is not None) and (sum_ztz_y is not None):
        X_ravel = X.ravel()
        v = scaled_hrf(a, t_r, n_times_atom)
        _grad = np.convolve(v, sum_ztz, 'valid') - sum_ztz_y
        cost = 0.5 * v.dot(_grad) + 0.5 * X_ravel.dot(X_ravel)
    else:  # full computation case (iteration on n_atoms)
        n_atoms, _ = z.shape
        v = scaled_hrf(a, t_r, n_times_atom)
        X_hat = np.zeros_like(X)
        for k in range(n_atoms):
            X_hat += np.outer(u[k, :], np.convolve(v, z[k, :]))
        residual = (X_hat - X).ravel()
        cost = 0.5 * residual.dot(residual)
    return cost


def _grad_v_scaled_hrf(a, t_r, n_times_atom, sum_ztz, sum_ztz_y=None):
    """ v d-basis HRF gradient.

    Parameters
    ----------
    a : array, shape (n_hrf_rois, n_param_HRF), init. HRF parameters
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    n_times_atom : int, (default=30), number of points on which represent the
        Haemodynamic Response Function (HRF), this leads to the duration of the
        response function, duration = n_times_atom * t_r

    Return
    ------
    grad : float, HRFs gradient
    """
    v = scaled_hrf(a, t_r, n_times_atom)
    delta_d_v = delta_derivative_double_gamma_hrf(a, t_r, n_times_atom)
    residual = np.convolve(v, sum_ztz, 'valid')
    if sum_ztz_y is not None:
        residual -= sum_ztz_y
    return delta_d_v.dot(residual)


def _grad_v_hrf_d_basis(a, AtA, AtX=None):
    """ v d-basis HRF gradient.

    Parameters
    ----------
    a : array, shape (n_hrf_rois, n_param_HRF), init. HRF parameters
    AtA : array, shape (n_atoms_hrf, n_atoms_hrf), precomputed operator
    AtX : array, shape (n_atoms_hrf), precomputed operator

    Return
    ------
    grad : array, shape (n_atoms_hrf, ), HRFs gradient
    """
    grad = a.dot(AtA)
    if AtX is not None:
        grad -= AtX
    return grad


def _grad_z(z, uvtuv, uvtX=None):
    """ z gradient.

    Parameters
    ----------
    z : array, shape (n_atoms, n_times_valid), temporal components
    uvtuv : array, shape (n_atoms, n_atoms, 2 * n_times_atom - 1), precomputed
        operator
    uvtX : array, shape (n_atoms, n_times_valid) computed operator image

    Return
    ------
    grad : array, shape (n_atoms, n_times_valid), temporal components gradient
    """
    grad = _compute_uvtuv_z(z=z, uvtuv=uvtuv)
    if uvtX is not None:
        grad -= uvtX
    return grad


def _grad_u_k(u, B, C, k, rois_idx):
    """ u gradient.

    Parameters
    ----------
    u : array, shape (n_atoms, n_voxels), spatial maps
    B : array, shape (n_atoms, n_voxels), precomputed operator
    C : array, shape (n_atoms, n_atoms), precomputed operator
    k : int, the index of the considered component
    rois_idx: array, shape (n_hrf_rois, max_indices_per_rois), HRF ROIs

    Return
    ------
    grad : array, shape (n_voxels), spatial maps gradient for one component
    """
    _, n_voxels = B[0, :, :].shape
    grad = np.empty(n_voxels)
    for m in range(rois_idx.shape[0]):
        indices = get_indices_from_roi(m, rois_idx)
        grad[indices] = C[m, k, :].dot(u[:, indices])
        grad[indices] -= B[m, k, indices]
    return grad


@numba.jit((numba.float64[:, :], numba.float64[:, :], numba.float64[:, :],
            numba.int64[:, :]), nopython=True, cache=True, fastmath=True)
def construct_X_hat_from_v(v, z, u, rois_idx):  # pragma: no cover
    """Return X_hat from v, z, u.

    Parameters
    ----------
    v : array, shape (n_hrf_rois, n_times_atom), HRFs
    z : array, shape (n_atoms, n_times_valid), temporal components
    u : array, shape (n_atoms, n_voxels), spatial maps
    rois_idx: array, shape (n_hrf_rois, max_indices_per_rois), HRF ROIs

    Return
    ------
    X_hat : array, shape (n_voxels, n_times), estimated fMRI data
    """
    # np.convolve construct_X_hat case
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
def construct_X_hat_from_H(H, z, u, rois_idx):  # pragma: no cover
    """Return X_hat from H, z, u.

    Parameters
    ----------
    H : array, shape (n_hrf_rois, n_times_valid, n_times), Toeplitz matrices
    z : array, shape (n_atoms, n_times_valid), temporal components
    u : array, shape (n_atoms, n_voxels), spatial maps
    rois_idx: array, shape (n_hrf_rois, max_indices_per_rois), HRF ROIs

    Return
    ------
    X_hat : array, shape (n_voxels, n_times), estimated fMRI data
    """
    # Toeplitz matrix construct_X_hat case
    n_voxels, n_times = u.shape[1], H.shape[1]
    zu = z.T.dot(u)
    X_hat = np.empty((n_voxels, n_times))
    for m in range(rois_idx.shape[0]):
        indices = get_indices_from_roi(m, rois_idx)
        X_hat[indices, :] = H[m, :, :].dot(zu[:, indices]).T
    return X_hat


def _obj(X, prox, u, z, rois_idx, H=None, v=None, valid=True, return_reg=True,
         lbda=None, rho=2.0, prox_z='tv'):
    """ Main objective function.

    Parameters
    ----------
    X : array, shape (n_voxels, n_times), fMRI data
    prox : func, proximal function on the spatial maps
    z : array, shape (n_atoms, n_times_valid), temporal components
    u : array, shape (n_atoms, n_voxels), spatial maps
    rois_idx: array, shape (n_hrf_rois, max_indices_per_rois), HRF ROIs
    H : array or None, (default=None), shape
        (n_hrf_rois, n_times_valid, n_times), Toeplitz matrices
    v : array or None, (default=None), shape (n_hrf_rois, n_times_atom), HRFs
    valid : bool, (default=True), whether or not to project the spatial maps
        into the valid subspace
    return_reg : bool, (default=True), whether or not to include the temporal
        regularization penalty in the cost-function value
    lbda : float or None, (default=None), the temporal regularization parameter
    rho : float, (default=2.0), the elastic-net temporal regularization
        parameter
    prox_z : str, (default='tv'), temporal proximal operator should be in
        ['tv', 'l1', 'l2', 'elastic-net']

    Return
    ------
    cost-function : float, the global cost-function value
    """
    u = np.r_[[prox(u_k) for u_k in u]] if valid else u

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
            if prox_z == 'tv':
                regu = lbda * np.sum(np.abs(np.diff(z, axis=-1)))
            if prox_z == 'l1':
                regu = lbda * np.sum(np.abs(z))
            if prox_z == 'l2':
                regu = lbda * np.linalg.norm(z)
            if prox_z == 'elastic-net':
                l1 = np.sum(np.abs(z))
                l2 = np.linalg.norm(z)
                regu = lbda * (l1 + rho / 2.0 * l2)
            return cost + regu
        else:
            raise ValueError("obj must have lbda to return regularization "
                             "value")
    else:
        return cost
