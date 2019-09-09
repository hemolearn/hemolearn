""" Simulated data module: provide function to generate simulated fMRI data
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>

import numpy as np

from .checks import check_random_state
from .hrf_model import spm_hrf
from .convolution import make_toeplitz
from .loss_grad import construct_X_hat_from_v
from .utils import add_gaussian_noise
from .atlas import split_atlas


def _gen_single_case_checkerboard(len_square, n=5, random_seed=None):
    """ Return a matrice of 0.0 with only one block one 1.0. """
    rng = check_random_state(random_seed)
    d = int(len_square / n)
    block_coor = (slice(d, 2*d), slice(d*(n-2), d*(n-1)))
    u = np.zeros((len_square, len_square))
    block_values = [1.0 + np.abs(0.5 * rng.randn()) for _ in range(d*d)]
    block_values = np.array(block_values).reshape((d, d))
    u[block_coor] = block_values
    return u


def _2_blocks_signal(n_times_valid=300, n=10, rng=np.random):
    """ return a 1d signal of n blocks of length T
    """
    d = int(n_times_valid / n)
    b_11 = slice(d, 2*d)
    b_12 = slice(d*(n-4), d*(n-3))
    b_21 = slice(3*d, 4*d)
    b_22 = slice(d*(n-2), d*(n-1))
    z_0 = np.zeros(n_times_valid)
    z_1 = np.zeros(n_times_valid)
    z_0[b_11] = 1.0 + np.abs(0.5 * rng.randn())
    z_0[b_12] = 1.0 + np.abs(0.5 * rng.randn())
    z_1[b_21] = 1.0 + np.abs(0.5 * rng.randn())
    z_1[b_22] = 1.0 + np.abs(0.5 * rng.randn())
    return z_0, z_1


def simulated_data(t_r=1.0, n_voxels=100, n_times_valid=100, n_times_atom=30, snr=1.0,
                   eta=10.0, random_seed=None):
    """ Generate simulated BOLD data with its temporal components z and the
    corresponding maps u.
    """
    rng = check_random_state(None)

    hrf_rois = {1: range(n_voxels)}
    rois_idx, _, _ = split_atlas(hrf_rois)

    z_0, z_1 = _2_blocks_signal(n_times_valid=n_times_valid, rng=rng)
    z = np.vstack([z_0, z_1])

    len_square = int(np.sqrt(n_voxels))
    u_0 = _gen_single_case_checkerboard(len_square, n=5, random_seed=rng)
    u_1 = u_0.T
    u_0 = u_0.flatten()[:, None]
    u_1 = u_1.flatten()[:, None]
    u_0 *= (eta / np.sum(np.abs(u_0)))
    u_1 *= (eta / np.sum(np.abs(u_1)))
    u = np.c_[u_0, u_1].T
    n_voxels = u_0.size

    v = spm_hrf(t_r=t_r, n_times_atom=n_times_atom)[None, :]

    X = construct_X_hat_from_v(v, z, u, rois_idx)
    noisy_X, _ = add_gaussian_noise(X, snr=snr, random_state=rng)

    return noisy_X, X, u, v, z, hrf_rois
