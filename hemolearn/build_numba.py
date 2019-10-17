"""Numba build module: compile each Numba functions of HemoLearn package"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

from .constants import _precompute_uvtuv, _precompute_d_basis_constant
from .convolution import adjconv_uv
from .prox import _prox_positive_l2_ball, _prox_l1_simplex
from .atlas import get_indices_from_roi
from .loss_grad import construct_X_hat_from_v, construct_X_hat_from_H
from .utils import _set_up_test


def build_numba_functions_of_hemolearn():
    """ Build each Numba functions of HemoLearn package.
    """
    kwargs = _set_up_test(0)
    X, u, z = kwargs['X'], kwargs['u'], kwargs['z']
    v, H, rois_idx = kwargs['v'], kwargs['H'], kwargs['rois_idx']
    u_k = u[0, :]
    uz = u.T.dot(z)
    residual_i = X
    m = 0
    _precompute_uvtuv(u, v, rois_idx)
    _precompute_d_basis_constant(X, uz, H)
    adjconv_uv(residual_i, u, v, rois_idx)
    _prox_positive_l2_ball(u_k, 1.0)
    _prox_l1_simplex(u_k, 10.0)
    get_indices_from_roi(m, rois_idx)
    construct_X_hat_from_v(v, z, u, rois_idx)
    construct_X_hat_from_H(H, z, u, rois_idx)
