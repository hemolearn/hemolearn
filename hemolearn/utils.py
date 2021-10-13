"""Utility module: gather various usefull functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import cProfile
import numpy as np
from scipy.signal import peak_widths, find_peaks

from .hrf_model import scaled_hrf
from .atlas import split_atlas
from .constants import _precompute_B_C
from .checks import check_random_state
from .convolution import make_toeplitz


def _compute_uvtuv_z(z, uvtuv):
    """ compute uvtuv_z

    Parameters
    ----------
    z : array, shape (n_atoms, n_times_valid) temporal components
    uvtuv : array, shape (n_atoms, n_atoms, 2 * n_times_atom - 1) precomputed
        operator

    Return
    ------
    uvtuv_z : array, shape (n_atoms, n_times_valid) computed operator image
    """
    n_atoms, n_times_valid = z.shape
    uvtuv_z = np.empty((n_atoms, n_times_valid))
    for k0 in range(n_atoms):
        _sum = np.convolve(z[0, :], uvtuv[k0, 0, :], mode='same')
        for k in range(1, n_atoms):
            _sum += np.convolve(z[k, :], uvtuv[k0, k, :], mode='same')
        uvtuv_z[k0, :] = _sum
    return uvtuv_z


def lipschitz_est(AtA, shape, nb_iter=30, tol=1.0e-6, verbose=False):
    """ Estimate the Lipschitz constant of the operator AtA.

    Parameters
    ----------
    AtA : func, the operator
    shape : tuple, the dimension variable space
    nb_iter : int, default(=30), the maximum number of iteration for the
        estimation
    tol : float, default(=1.0e-6), the tolerance to do early stopping
    verbose : bool, default(=False), the verbose

    Return
    ------
    L : float, Lipschitz constant of the operator
    """
    x_old = np.random.randn(*shape)
    converge = False
    for _ in range(nb_iter):
        x_new = AtA(x_old) / np.linalg.norm(x_old)
        if(np.abs(np.linalg.norm(x_new) - np.linalg.norm(x_old)) < tol):
            converge = True
            break
        x_old = x_new
    if not converge and verbose:
        print("Spectral radius estimation did not converge")
    return np.linalg.norm(x_new)


def fwhm(t_r, hrf):  # pragma: no cover
    """Return the full width at half maximum of the HRF.

    Parameters
    ----------
    t : array, shape (n_times_atom, ), the time
    hrf : array, shape (n_times_atom, ), HRF

    Return
    ------
    s : float, the full width at half maximum of the HRF
    """
    peaks_in_idx, _ = find_peaks(hrf)
    fwhm_in_idx, _, _, _ = peak_widths(hrf, peaks_in_idx, rel_height=0.5)
    fwhm_in_idx = fwhm_in_idx[0]  # catch the first (and only) peak
    return t_r * int(fwhm_in_idx)  # in seconds


def tp(t_r, hrf):  # pragma: no cover
    """ Return time to peak oh the signal of the HRF.

    Parameters
    ----------
    t : array, shape (n_times_atom, ), the time
    hrf : array, shape (n_times_atom, ), HRF

    Return
    ------
    s : float, time to peak oh the signal of the HRF
    """
    n_times_atom = len(hrf)
    t = np.linspace(0.0, t_r * n_times_atom, n_times_atom)
    return t[np.argmax(hrf)]  # in seconds


def add_gaussian_noise(signal, snr, random_state=None):  # pragma: no cover
    """ Add a Gaussian noise to inout signal to output a noisy signal with the
    targeted SNR.

    Parameters
    ----------
    signal : array, the given signal on which add a Guassian noise.
    snr : float, the expected SNR for the output signal.
    random_state :  int or None (default=None),
        Whether to impose a seed on the random generation or not (for
        reproductability).

    Return
    ------
    noisy_signal : array, the noisy produced signal.
    noise : array, the additif produced noise.
    """
    # draw the noise
    rng = check_random_state(random_state)
    s_shape = signal.shape
    noise = rng.randn(*s_shape)
    # adjuste the standard deviation of the noise
    true_snr_num = np.linalg.norm(signal)
    true_snr_deno = np.linalg.norm(noise)
    true_snr = true_snr_num / (true_snr_deno + np.finfo(np.float).eps)
    std_dev = (1.0 / np.sqrt(10**(snr/10.0))) * true_snr
    noise = std_dev * noise
    noisy_signal = signal + noise

    return noisy_signal, noise


def sort_by_expl_var(u, z, v, hrf_rois):  # pragma: no cover
    """ Sorted the temporal the spatial maps and the associated activation by
    explained variance.

    Parameters
    ----------
    u : array, shape (n_atoms, n_voxels), spatial maps
    z : array, shape (n_atoms, n_times_valid), temporal components
    v : array, shape (n_hrf_rois, n_times_atom), HRFs
    hrf_rois : dict (key: ROIs labels, value: indices of voxels of the ROI)
        atlas HRF

    Return
    ------
    u : array, shape (n_atoms, n_voxels), the order spatial maps
    z : array, shape (n_atoms, n_times_valid), the order temporal components
    variances : array, shape (n_atoms, ) the order variances for each
        components
    """
    rois_idx, _, n_hrf_rois = split_atlas(hrf_rois)
    n_atoms, n_voxels = u.shape
    _, n_times_valid = z.shape
    n_hrf_rois, n_times_atom = v.shape
    n_times = n_times_valid + n_times_atom - 1
    variances = np.empty(n_atoms)
    # recompose each X_hat_k (components) to compute its variance
    for k in range(n_atoms):
        X_hat = np.empty((n_voxels, n_times))
        for m in range(n_hrf_rois):  # iteration on all HRF ROIs
            # get voxels idx for this ROI
            voxels_idx = rois_idx[m, 1:rois_idx[m, 0]]
            # iterate on each voxels
            for j in voxels_idx:
                X_hat[j, :] = np.convolve(u[k, j] * v[m, :], z[k, :])
        variances[k] = np.var(X_hat)
    order = np.argsort(variances)[::-1]
    return u[order, :], z[order, :], variances[order]


def _set_up_test(seed):  # pragma: no cover
    """ General set up function for the tests.

    Parameters
    ----------
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the analysis

    Return
    ------
    kwargs : dict, the setting to make unitary tests the keys are (t_r,
        n_hrf_rois, n_atoms, n_voxels, n_times, n_times_atom, n_times_valid,
        rois_idx, X, z, u, H, v, B, C)
    """
    rng = check_random_state(None)
    t_r = 1.0
    n_hrf_rois = 2
    n_atoms, n_voxels = 4, 200
    n_times, n_times_atom = 100, 20
    n_times_valid = n_times - n_times_atom + 1
    labels = np.arange(n_hrf_rois, dtype=int)
    indices = np.split(np.arange(n_voxels, dtype=int), n_hrf_rois)
    hrf_rois = dict(zip(labels, indices))
    rois_idx, _, n_hrf_rois = split_atlas(hrf_rois)
    X = rng.randn(n_voxels, n_times)
    u = rng.randn(n_atoms, n_voxels)
    z = rng.randn(n_atoms, n_times_valid)
    v = np.r_[[scaled_hrf(delta=1.0, t_r=t_r, n_times_atom=n_times_atom)
               for _ in range(n_hrf_rois)]]
    H = np.empty((n_hrf_rois, n_times, n_times_valid))
    for m in range(n_hrf_rois):
        H[m, ...] = make_toeplitz(v[m], n_times_valid)
    B, C = _precompute_B_C(X, z, H, rois_idx)
    # gather the various parameters in a big dictionary for unit-tests
    kwargs = dict(t_r=t_r, n_hrf_rois=n_hrf_rois, n_atoms=n_atoms,
                  n_voxels=n_voxels, n_times=n_times, rng=rng,
                  n_times_atom=n_times_atom, n_times_valid=n_times_valid,
                  rois_idx=rois_idx, X=X, z=z, u=u, H=H, v=v, B=B, C=C)
    return kwargs


def th(x, t, absolute=True):  # pragma: no cover
    """Return threshold level to retain t entries of array x."""
    if isinstance(t, str):
        t = float(t[:-1]) / 100.
    elif isinstance(t, float):
        pass
    else:
        raise ValueError(f"t (={t}) type not understtod: shoud be a float"
                         f" between 0 and 1 or a string such as '80%'")
    n = -int(t * len(x.flatten()))
    if absolute:
        return np.sort(np.abs(x.flatten()))[n]
    else:
        return np.sort(x.flatten())[n]


def profile_me(func):  # pragma: no cover
    """ Profiling decorator, produce a report <func-name>.profile to be open as
    `python -m snakeviz  <func-name>.profile`

    Parameters
    ----------
    func : func, function to profile
    """
    def profiled_func(*args, **kwargs):
        filename = func.__name__ + '.profile'
        prof = cProfile.Profile()
        ret = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(filename)
        return ret
    return profiled_func
