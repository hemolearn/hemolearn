"""HRF model module: Hemodynamic Responses Function (HRF) models"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
from scipy.stats import gamma

from .checks import check_len_hrf


MIN_DELTA = 0.5
MAX_DELTA = 2.0


def spm_hrf_2_basis(t_r, n_times_atom):
    """ SPM HRF 2 basis function.

    Parameters
    ----------
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    n_atoms : int, number of components on which to decompose the neural
        activity (number of temporal components and its associated spatial
        maps)

    Return
    ------
    hrf_basis : array, shape (n_hrf_atoms, n_times_atom), HRF basis
    """
    return _spm_hrf_d_basis(2, t_r, n_times_atom)


def spm_hrf_3_basis(t_r, n_times_atom):
    """ SPM HRF 3 basis function.

    Parameters
    ----------
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    n_atoms : int, number of components on which to decompose the neural
        activity (number of temporal components and its associated spatial
        maps)

    Return
    ------
    hrf_basis : array, shape (n_hrf_atoms, n_times_atom), HRF basis
    """
    return _spm_hrf_d_basis(3, t_r, n_times_atom)


def _spm_hrf_d_basis(d, t_r, n_times_atom):
    """ Private helper to define the SPM HRF 2/3 basis function.

    Parameters
    ----------
    d : int, the number of atoms in the HRF basis, possible values are 2 or 3
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    n_hrf_atoms : int, number of components on which to decompose the neural
        activity (number of temporal components and its associated spatial
        maps).

    Return
    ------
    hrf_basis : array, shape (n_hrf_atoms, n_times_atom), HRF basis
    """
    assert d in [2, 3], "HRF basis can only have 2 or 3 atoms"

    dur = t_r * n_times_atom

    h_1 = _spm_hrf(delta=1.0, t_r=t_r, dur=dur)[0]
    h_2_ = _spm_hrf(delta=1.0, t_r=t_r, dur=dur, onset=0.0)[0]
    h_2__ = _spm_hrf(delta=1.0, t_r=t_r, dur=dur, onset=t_r)[0]
    h_3_ = _spm_hrf(delta=1.0, t_r=t_r, dur=dur, p_disp=1.001)[0]

    h_2 = h_2_ - h_2__
    h_3 = (h_1 - h_3_) / 0.001

    h_1 = check_len_hrf(h_1, n_times_atom)
    h_2 = check_len_hrf(h_2, n_times_atom)
    h_3 = check_len_hrf(h_3, n_times_atom)

    if d == 2:
        return np.c_[h_1, h_2].T
    else:
        return np.c_[h_1, h_2, h_3].T


def spm_scaled_hrf(delta, t_r, n_times_atom):
    """ SPM scaled HRF.

    Parameters
    ----------
    delta : float, temporal dilation to pilot the HRF inflation
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    n_times_atom : int, number of components on which to decompose the neural
        activity (number of temporal components and its associated spatial
        maps).

    Return
    ------
    hrf : array, shape (n_times_atom, ), HRF
    """
    _hrf = _spm_hrf(delta=delta, t_r=t_r, dur=n_times_atom * t_r)[0]
    return check_len_hrf(_hrf, n_times_atom)


def spm_hrf(t_r, n_times_atom):
    """ SPM HRF.

    Parameters
    ----------
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    n_times_atom : int, number of components on which to decompose the neural
        activity (number of temporal components and its associated spatial
        maps).

    Return
    ------
    hrf : array, shape (n_times_atom, ), HRF
    """
    _hrf = _spm_hrf(delta=1.0, t_r=t_r, dur=n_times_atom * t_r)[0]
    return check_len_hrf(_hrf, n_times_atom)


def _spm_hrf(delta, t_r=1.0, dur=60.0, dt=0.001,
             p_delay=6, undershoot=16.0, p_disp=1.0, u_disp=1.0,
             p_u_ratio=0.167, onset=0.0):
    """ Private helper for SPM scaled HRF.

    Parameters
    ----------
    delta : float, temporal dilation to pilot the HRF inflation
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    dur : int, (default=60.0), the time duration on which to represent the HRF
    dt : float, (default=0.001), the temporal resolution before subsampling
    p_delay : int, (default=6), steepness of the peak
    undershoot : float, (default=16.0), undershoot depth
    p_disp : float, (default=1.0), width of the peak
    u_disp : float, (default=1.0), width of the undershoot
    p_u_ratio : float, (default=0.167), ratio  of the peak w.r.t the undershoot
    onset : float, (default=0.0), onset of the HRF

    Return
    ------
    hrf : array, shape (dur / t_r, ), HRF
    """
    if (delta < MIN_DELTA) or (delta > MAX_DELTA):
        raise ValueError("delta should belong in [{0}, {1}]; wich correspond"
                         " to a max FWHM of 10.52s and a min FWHM of 2.80s"
                         ", got delta = {2}".format(MIN_DELTA, MAX_DELTA,
                                                    delta))

    # dur: the (continious) time segment on which we represent all
    # the HRF. Can cut the HRF too early. The time scale is second.
    t = np.linspace(0, dur, int(float(dur) / dt)) - float(onset) / dt
    scaled_time_stamps = delta * t

    peak = gamma.pdf(scaled_time_stamps, p_delay/p_disp, loc=dt/p_disp)
    undershoot = gamma.pdf(scaled_time_stamps, undershoot/u_disp,
                           loc=dt/u_disp)
    hrf = peak - p_u_ratio * undershoot

    return hrf[::int(t_r/dt)], t[::int(t_r/dt)]
