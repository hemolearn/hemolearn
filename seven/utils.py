"""Utility module"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import cProfile
from datetime import datetime
import numpy as np
from scipy.interpolate import splrep, sproot
from nilearn import input_data

from .hrf_model import spm_hrf
from .atlas import split_atlas
from .constants import _precompute_B_C
from .checks import check_random_state
from .convolution import make_toeplitz


def _compute_uvtuv_z(z, uvtuv):
    """ compute uvtuv_z

    Parameters
    ----------
    z : array, shape (n_atoms, n_times_valid)
    uvtuv : array, shape (n_atoms, n_atoms, 2 * n_times_atom - 1)

    Return
    ------
    uvtuv_z : array, shape (n_atoms, n_times_valid)
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
    """ Est. the cst Lipschitz of AtA. """
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


def fwhm(t, hrf, k=3):
    """Return the full width at half maximum. """
    half_max = np.amax(hrf) / 2.0
    s = splrep(t, hrf - half_max, k=k)
    roots = sproot(s)
    try:
        return np.abs(roots[1] - roots[0])
    except IndexError:
        return -1


def tp(t, hrf):
    """ Return time to peak oh the signal. """
    return t[np.argmax(hrf)]


def get_nifti_ext(func_fname):
    """ Return the extension of the Nifti. """
    err_msg = "Filename extension not understood, {extension}"
    fname, ext = os.path.splitext(func_fname)
    if ext == '.gz':
        fname, ext_ = os.path.splitext(fname)
        if ext_ == '.nii':
            return fname, ext_ + ext
        else:
            raise ValueError(err_msg.format(extension=ext))
    elif ext == '.nii':
        return fname, ext
    else:
        raise ValueError(err_msg.format(extension=ext))


def fmri_preprocess(
        func_fname, mask_img=None, sessions=None, smoothing_fwhm=None,
        standardize=False, detrend=False, low_pass=None, high_pass=None,
        t_r=None, target_affine=None, target_shape=None,
        mask_strategy='background', mask_args=None, sample_mask=None,
        dtype=None, memory_level=1, memory=None, verbose=0,
        confounds=None, suffix='_preproc'):
    """ Preprocess the fMRI data. """
    if not isinstance(func_fname, str):
        raise ValueError("func_fname should be the filename of "
                         "a 4d Nifti file")

    masker = input_data.NiftiMasker(
        mask_img=mask_img, sessions=sessions, smoothing_fwhm=smoothing_fwhm,
        standardize=standardize, detrend=detrend, low_pass=low_pass,
        high_pass=high_pass, t_r=t_r, target_affine=target_affine,
        target_shape=target_shape, mask_strategy=mask_strategy,
        mask_args=mask_args, sample_mask=sample_mask, dtype=dtype,
        memory_level=memory_level, memory=memory, verbose=verbose)

    preproc_X = masker.fit_transform(func_fname, confounds=confounds)
    preproc_X_img = masker.inverse_transform(preproc_X)
    fname, ext = get_nifti_ext(func_fname)
    nfname = fname + suffix + ext
    preproc_X_img.to_filename(nfname)

    return nfname


def sort_atoms_by_explained_variances(u, z, v, hrf_rois):
    """ Sorted the temporal the spatial maps and the associated activation by
    explained variance."""
    n_atoms, n_voxels = u.shape
    _, n_times_valid = z.shape
    n_hrf_rois, n_times_atom = v.shape
    n_times = n_times_valid + n_times_atom - 1
    variances = np.empty(n_atoms)
    for k in range(n_atoms):
        X_hat = np.empty((n_voxels, n_times))
        for m in range(n_hrf_rois):
            for j in hrf_rois[m + 1]:
                X_hat[j, :] = np.convolve(u[k, j] * v[m, :], z[k, :])
        variances[k] = np.var(X_hat)
    order = np.argsort(variances)[::-1]
    return u[order, :], z[order, :], variances[order]


def get_unique_dirname(prefix):
    """ Return a unique dirname based on the time and the date."""
    msg = "prefix should be a string, got {}".format(prefix)
    assert isinstance(prefix, str), msg
    date = datetime.now()
    date_tag = '{0}{1:02d}{2:02d}{3:02d}{4:02d}{5:02d}'.format(
                                        date.year, date.month, date.day,
                                        date.hour, date.minute, date.second)
    return prefix + date_tag


def _set_up_test(seed):
    """ General set up function for the tests. """
    rng = check_random_state(None)
    t_r = 1.0
    n_hrf_rois = 2
    n_atoms, n_voxels = 4, 100
    n_times, n_times_atom = 100, 20
    n_times_valid = n_times - n_times_atom + 1
    labels = np.arange(n_hrf_rois, dtype=int)
    indices = np.split(np.arange(n_voxels, dtype=int), n_hrf_rois)
    hrf_rois = dict(zip(labels, indices))
    rois_idx, _, n_hrf_rois = split_atlas(hrf_rois)
    X = rng.randn(n_voxels, n_times)
    u = rng.randn(n_atoms, n_voxels)
    z = rng.randn(n_atoms, n_times_valid)
    v = np.r_[[spm_hrf(t_r=t_r, n_times_atom=n_times_atom)
               for _ in range(n_hrf_rois)]]
    H = np.empty((n_hrf_rois, n_times, n_times_valid))
    for m in range(n_hrf_rois):
        H[m, ...] = make_toeplitz(v[m], n_times_valid)
    B, C = _precompute_B_C(X, z, H, rois_idx)
    kwargs = dict(t_r=t_r, n_hrf_rois=n_hrf_rois, n_atoms=n_atoms,
                  n_voxels=n_voxels, n_times=n_times,
                  n_times_atom=n_times_atom, n_times_valid=n_times_valid,
                  rois_idx=rois_idx, X=X, z=z, u=u, H=H, v=v, B=B, C=C)
    return kwargs


def profile_me(fn):
    """ Profiling decorator. """
    def profiled_fn(*args, **kwargs):
        filename = fn.__name__ + '.profile'
        prof = cProfile.Profile()
        ret = prof.runcall(fn, *args, **kwargs)
        prof.dump_stats(filename)
        return ret
    return profiled_fn
