"""Utility module: gather various usefull functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import cProfile
from datetime import datetime
import numpy as np
from scipy.signal import peak_widths, find_peaks
from nilearn import input_data

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


def fwhm(t_r, hrf):
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


def tp(t_r, hrf):
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


def get_nifti_ext(func_fname):
    """ Return the extension of the Nifti.

    Parameters
    ----------
    func_fname : str, the filename of the fMRI data

    Return
    ------
    fname : str, the basename of the fMRI data
    ext : str, the extension of the fMRI data
    """
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


def fmri_preprocess(  # pragma: no cover
        func_fname, mask_img=None, sessions=None, smoothing_fwhm=None,
        standardize=False, detrend=False, low_pass=None, high_pass=None,
        t_r=None, target_affine=None, target_shape=None,
        mask_strategy='background', mask_args=None, sample_mask=None,
        dtype=None, memory_level=1, memory=None, verbose=0,
        confounds=None, preproc_fname=None):
    """ Preprocess the fMRI data.

    Parameters
    ----------
    func_fname : str, the filename of the fMRI data
    mask_img : Nifti-like img or None, (default=None),
    sessions : array or None, (default=None), add a session level to the
        preprocessing
    smoothing_fwhm : float or None, (default=None), if smoothing_fwhm is not
        None, it gives the full-width half maximum in millimeters of the
        spatial smoothing to apply to the signal.
    standardize : bool, (default=False), if standardize is True, the
        time-series are centered and normed: their mean is put to 0 and their
        variance to 1 in the time dimension.
    detrend : bool, (defaul==False), whether or not to detrend the BOLD signal
    low_pass : float or None, (defaul==None), the limit low freq pass band to
        clean the BOLD signal
    high_pass : float or None, (default=None), the limit high freq pass band to
        clean the BOLD signal
    t_r : float or None, (default=None), Time of Repetition, fMRI acquisition
        parameter, the temporal resolution
    target_affine :  3x3 or 4x4 array or None, (default=None), this parameter
        is passed to nilearn.image.resample_img
    target_shape : 3-tuple of integers or None, (default=None),this parameter
        is passed to nilearn.image.resample_img
    mask_strategy : {‘background’, ‘epi’ or ‘template’},
        (default='background'), the strategy used to compute the mask: use
        ‘background’ if your images present a clear homogeneous background,
        ‘epi’ if they are raw EPI images, or you could use ‘template’ which
        will extract the gray matter part of your data by resampling the MNI152
        brain mask for your data’s field of view. Depending on this value, the
        mask will be computed from nilearn.masking.compute_background_mask,
        masking.compute_epi_mask or nilearn.masking.compute_gray_matter_mask.
    mask_args : dict, (default=None), if mask is None, these are additional
        parameters passed to masking.compute_background_mask or
        nilearn.masking.compute_epi_mask to fine-tune mask computation. Please
        see the related documentation for details.
    sample_mask : Any type compatible with numpy-array indexing,
        (default=None), any type compatible with numpy-array indexing Masks the
        niimgs along time/fourth dimension. This complements 3D masking by the
        mask_img argument. This masking step is applied before data
        preprocessing at the beginning of NiftiMasker.transform. This is useful
        to perform data subselection as part of a scikit-learn pipeline.
    dtype :  {dtype, “auto”} or None, (default=None), data type toward which
        the data should be converted. If “auto”, the data will be converted to
        int32 if dtype is discrete and float32 if it is continuous.
    memory : instance of joblib.Memory or str, (default=None), used to cache
        the masking process. By default, no caching is done. If a string is
        given, it is the path to the caching directory.
    memory_level : int, (default=1), rough estimator of the amount of memory
        used by caching. Higher value means more memory for caching
    verbose : int, (default=0), indicate the level of verbosity. By default,
        nothing is printed
    confounds : Nifti-like img or None, (default=None), list of confounds (2D
        arrays or filenames pointing to CSV files). Must be of same length than
        imgs_list.
    preproc_fname : str or None, (default=None), the full filename of the
        preprocessed fMRI data filename, if not given simple '_preproc' suffix
        is add to the original filename

    Return
    ------
    nfname : str, the filename of the preprocessed fMRI data
    nfname : array, the array of the preprocessed fMRI data
    nfname : nifti data, the nifti data of the preprocessed fMRI data

    """
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
    if preproc_fname is None:
        fname, ext = get_nifti_ext(func_fname)
        preproc_fname = fname + '_preproc' + ext
    preproc_X_img.to_filename(preproc_fname)

    return preproc_fname, preproc_X_img, preproc_X


def add_gaussian_noise(signal, snr, random_state=None):
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


def sort_atoms_by_explained_variances(u, z, v, hrf_rois):
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
    n_atoms, n_voxels = u.shape
    _, n_times_valid = z.shape
    n_hrf_rois, n_times_atom = v.shape
    n_times = n_times_valid + n_times_atom - 1
    variances = np.empty(n_atoms)
    # recompose each X_hat_k (components) to compute its variance
    for k in range(n_atoms):
        X_hat = np.empty((n_voxels, n_times))
        for m in range(n_hrf_rois):
            for j in hrf_rois[m + 1]:
                X_hat[j, :] = np.convolve(u[k, j] * v[m, :], z[k, :])
        variances[k] = np.var(X_hat)
    order = np.argsort(variances)[::-1]
    return u[order, :], z[order, :], variances[order]


def get_unique_dirname(prefix):
    """ Return a unique dirname based on the time and the date.

    Parameters
    ----------
    prefix : str, the prefix to add to the directory name

    Return
    ------
    dirname : str, the unique directory name
    """
    msg = "prefix should be a string, got {}".format(prefix)
    assert isinstance(prefix, str), msg
    date = datetime.now()
    date_tag = '{0}{1:02d}{2:02d}{3:02d}{4:02d}{5:02d}'.format(
                                        date.year, date.month, date.day,
                                        date.hour, date.minute, date.second)
    return prefix + date_tag


def _set_up_test(seed):
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
    v = np.r_[[scaled_hrf(delta=1.0, t_r=t_r, n_times_atom=n_times_atom)
               for _ in range(n_hrf_rois)]]
    H = np.empty((n_hrf_rois, n_times, n_times_valid))
    for m in range(n_hrf_rois):
        H[m, ...] = make_toeplitz(v[m], n_times_valid)
    B, C = _precompute_B_C(X, z, H, rois_idx)
    # gather the various parameters in a big dictionary for unit-tests
    kwargs = dict(t_r=t_r, n_hrf_rois=n_hrf_rois, n_atoms=n_atoms,
                  n_voxels=n_voxels, n_times=n_times,
                  n_times_atom=n_times_atom, n_times_valid=n_times_valid,
                  rois_idx=rois_idx, X=X, z=z, u=u, H=H, v=v, B=B, C=C)
    return kwargs


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
