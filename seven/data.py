"""Synthetic data generation module"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
from skimage.draw import circle

from alphacsc.utils.hrf_model import spm_hrf
from alphacsc.datasets.fmri import add_gaussian_noise
from alphacsc.utils.compute_constants import (_compute_DtD_uv, compute_ztX,
                                              compute_ztz)
from alphacsc.utils.convolution import _dense_tr_conv_d


def _gen_one_voxel(tr, len_h, snr, s, T, rng):
    """Generate one synthetic voxel."""
    h = spm_hrf(tr, len_h)
    n_s = int(s * T)
    idx = np.arange(T)
    rng.shuffle(idx)
    idx = idx[: n_s]
    z = np.zeros(T)
    z[idx] = rng.randn(n_s)
    x = np.cumsum(z)
    z = np.diff(x)
    y = np.convolve(h, x)
    noisy_y, _, _ = add_gaussian_noise(y, snr=snr, random_state=rng)

    return noisy_y, y, x, z, h


def _gen_multi_voxels(tr, n_times_atom, snr, s, n_times_valid, n_atoms,
                      n_channels, rng):
    """Generate multiple synthetic voxel."""
    n_s = int(s * n_times_valid)  # l0-norm support of each z_k
    h = spm_hrf(tr, n_times_atom)  # HRF

    # construction of the synthetics signals
    l_z_k, l_Lz_k, l_h_conv_Lz_k, l_u_k = [], [], [], []
    for _ in range(n_atoms):
        # z_k should be s-sparse, we randomly choose the support
        idx_t = np.arange(n_times_valid)
        rng.shuffle(idx_t)
        idx_t = idx_t[: n_s]

        # declare the variables
        p_ = int(np.sqrt(n_channels))
        u_k = np.zeros((p_, p_))
        z_k = np.zeros(n_times_valid)

        # populate the non-zeros entries
        z_k[idx_t] = rng.randn(n_s)
        c = (rng.randint(p_), rng.randint(p_))
        r = int(p_ / 10)
        raw_, col_ = circle(c[0], c[1], r)
        u_k[raw_ % p_, col_ % p_] = 1.0
        u_k = (u_k / np.linalg.norm(u_k)).flatten()

        # construct the signals
        Lz_k = np.cumsum(z_k)
        z_k = np.diff(Lz_k)
        h_conv_Lz_k = np.convolve(h, Lz_k)

        # store the signals
        l_u_k.append(u_k)
        l_h_conv_Lz_k.append(h_conv_Lz_k)
        l_Lz_k.append(Lz_k)
        l_z_k.append(z_k)

    # concatenate the signals in matrix variables
    u = np.vstack(l_u_k)
    uv = np.c_[u, np.repeat(h[None, :], n_atoms, axis=0).squeeze()]
    h_conv_Lz = np.vstack(l_h_conv_Lz_k)
    Lz = np.vstack(l_Lz_k)
    z = np.vstack(l_z_k)

    # construct the ground-truth and the observed data
    X = np.dot(u.T, h_conv_Lz)
    noisy_X, _, _ = add_gaussian_noise(X, snr=snr, random_state=rng)

    DtD = _compute_DtD_uv(uv, n_channels)
    DtX = _dense_tr_conv_d(X, D=uv, n_channels=n_channels)
    LztX = compute_ztX(Lz[None, :, :], X[None, :, :])
    LztLz = compute_ztz(Lz[None, :, :], n_times_atom)

    return noisy_X, X, Lz, z, uv, u, h, DtD, DtX, LztLz, LztX
