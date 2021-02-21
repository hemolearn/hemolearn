""" Simulated data module: provide function to generate simulated fMRI data
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>

import numpy as np

from .checks import check_random_state
from .hrf_model import scaled_hrf
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


def _2_blocks_task_signal(n_times_valid=300, n1=10, n2=10, rng=np.random):
    """ return a 1d signal of n blocks of length T
    """
    d1 = int(n_times_valid / n1)
    d2 = int(n_times_valid / n2)
    b_11 = slice(d1, 2 * d1)
    b_12 = slice(d2 * (n2 - 4), d2 * (n2 - 3))
    b_21 = slice(3 * d1, 4 * d1)
    b_22 = slice(d2 * (n2 - 2), d2 * (n2 - 1))
    z_0 = np.zeros(n_times_valid)
    z_1 = np.zeros(n_times_valid)
    z_0[b_11] = 1.0 + np.abs(0.5 * rng.randn())
    z_0[b_12] = 1.0 + np.abs(0.5 * rng.randn())
    z_1[b_21] = 1.0 + np.abs(0.5 * rng.randn())
    z_1[b_22] = 1.0 + np.abs(0.5 * rng.randn())
    return z_0, z_1


def _2_blocks_rest_signal(n_times_valid=300, s=0.05, rng=np.random):
    """ return a 1d signal of n blocks of length T
    """
    Dz_0 = np.zeros(n_times_valid)
    Dz_1 = np.zeros(n_times_valid)
    idx_dirac_Dz_0 = rng.randint(0, n_times_valid, int(n_times_valid * s))
    ampl_dirac_Dz_0 = rng.randn(int(n_times_valid * s))
    Dz_0[idx_dirac_Dz_0] = ampl_dirac_Dz_0
    idx_dirac_Dz_1 = rng.randint(0, n_times_valid, int(n_times_valid * s))
    ampl_dirac_Dz_1 = rng.randn(int(n_times_valid * s))
    Dz_1[idx_dirac_Dz_1] = ampl_dirac_Dz_1
    z_0 = np.cumsum(Dz_0)
    z_1 = np.cumsum(Dz_1)
    z_0 -= np.mean(z_0)
    z_0 /= np.max(np.abs(z_0))
    z_1 -= np.mean(z_0)
    z_1 /= np.max(np.abs(z_1))
    return z_0, z_1


def simulated_data(t_r=1.0, n_voxels=100, n_times_valid=100, n_times_atom=30,
                   snr=1.0, eta=10.0, delta=1.0, z_type='rest', s=0.05, n1=10,
                   n2=10, random_seed=None):
    """ Generate simulated BOLD data with its temporal components z and the
    corresponding maps u.
    """
    rng = check_random_state(random_seed)

    hrf_rois = {1: range(n_voxels)}
    rois_idx, _, _ = split_atlas(hrf_rois)

    if z_type == 'rest':
        z_0, z_1 = _2_blocks_rest_signal(n_times_valid=n_times_valid, s=s,
                                         rng=rng)
    elif z_type == 'task':
        z_0, z_1 = _2_blocks_task_signal(n_times_valid=n_times_valid, n1=n1,
                                         n2=n2, rng=rng)
    elif 'rest&task':
        z_0, _ = _2_blocks_rest_signal(n_times_valid=n_times_valid, s=s,
                                       rng=rng)
        _, z_1 = _2_blocks_task_signal(n_times_valid=n_times_valid, n1=n1,
                                       n2=n2, rng=rng)
    else:
        raise ValueError("z_type should belong to "
                         "['rest', 'task'], got {}".format(z_type))
    z = np.vstack([z_0, z_1])

    len_square = int(np.sqrt(n_voxels))
    u_0 = _gen_single_case_checkerboard(len_square, n=4, random_seed=rng)
    u_1 = u_0.T
    u_0 = u_0.flatten()[:, None]
    u_1 = u_1.flatten()[:, None]
    u_0 *= (eta / np.sum(np.abs(u_0)))
    u_1 *= (eta / np.sum(np.abs(u_1)))
    u = np.c_[u_0, u_1].T
    n_voxels = u_0.size

    v = scaled_hrf(delta=delta, t_r=t_r, n_times_atom=n_times_atom)[None, :]

    X = construct_X_hat_from_v(v, z, u, rois_idx)
    noisy_X, _ = add_gaussian_noise(X, snr=snr, random_state=rng)

    return noisy_X, X, u, v, z, hrf_rois
