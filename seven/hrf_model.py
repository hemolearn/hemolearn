"""HRF model module: Hemodynamic Responses Function (HRF) models"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
from scipy.special import gammaln, xlogy

from .checks import check_len_hrf


# feasable scaled HRF domain
MIN_DELTA = 0.5
MAX_DELTA = 2.0

# double gamma HRF model constants
DT = 0.001
P_DELAY = 6.0
UNDERSHOOT = 16.0
P_DISP = 1.0
U_DISP = 1.0
P_U_RATIO = 0.167

# usefull precomputed HRF peak constants
LOC_PEAK = DT / P_DISP
A_1_PEAK = P_DELAY / P_DISP - 1
GAMMA_LN_A_PEAK = gammaln(P_DELAY / 1.0)

# usefull precomputed HRF undershoot constants
LOC_U = DT / U_DISP
A_1_U = UNDERSHOOT / U_DISP - 1
GAMMA_LN_A_U = gammaln(UNDERSHOOT / 1.0)


# helpers


def _gamma_pdf(x, a, loc=0.0, scale=1.0):
    """ Probability density function of the Gamma distribution.

    Parameters
    ----------
    x : float, quantiles
    a : float, normalized Gamma pdf parameter
    loc : float, (default=0) location parameter
    scale : float, (default=1) scale parameter

    Return
    ------
    p : float, probability density function evaluated at x
    """
    x = (x - loc) / scale
    support = x > 0.0
    x_valid = x[support]
    p = np.zeros_like(x)
    p[support] = np.exp(xlogy(a - 1, x_valid) - x_valid - gammaln(a))
    return p


def _gamma_pdf_hrf_peak(x):
    """ Precomputed gamma pdf for HRF peak (double gamma HRF model).

    Parameters
    ----------
    x : float, quantiles

    Return
    ------
    p : float, probability density function evaluated at x
    """
    x = np.copy(x)
    x -= LOC_PEAK
    support = x > 0.0
    x_valid = x[support]
    p = np.zeros_like(x)
    p[support] = np.exp(xlogy(A_1_PEAK, x_valid) - x_valid - GAMMA_LN_A_PEAK)
    return p


def _gamma_pdf_hrf_undershoot(x):
    """ Precomputed gamma pdf for HRF undershoot (double gamma HRF model).

    Parameters
    ----------
    x : float, quantiles

    Return
    ------
    p : float, probability density function evaluated at x
    """
    x = np.copy(x)
    x -= LOC_U
    support = x > 0.0
    x_valid = x[support]
    p = np.zeros_like(x)
    p[support] = np.exp(xlogy(A_1_U, x_valid) - x_valid - GAMMA_LN_A_U)
    return p


def _double_gamma_hrf(delta, t_r=1.0, dur=60.0, onset=0.0):
    """ Double Gamma HRF model.

    From Nistats package
   https://github.com/nistats/nistats/blob/master/nistats/hemodynamic_models.py

    Parameters
    ----------
    delta : float, temporal dilation to pilot the HRF inflation
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    dur : float, (default=60.0), the time duration on which to represent the
        HRF
    onset : float, (default=0.0), onset of the HRF

    Return
    ------
    hrf : array, shape (dur / t_r, ), HRF
    """
    # dur: the (continious) time segment on which we represent all
    # the HRF. Can cut the HRF too early. The time scale is second.
    t = np.linspace(0, dur, int(float(dur) / DT)) - float(onset) / DT
    t = t[::int(t_r/DT)]

    peak = _gamma_pdf_hrf_peak(delta * t)
    undershoot = _gamma_pdf_hrf_undershoot(delta * t)
    hrf = peak - P_U_RATIO * undershoot

    return hrf, t


def _delta_derivative_double_gamma_hrf(delta, t_r=1.0, dur=60.0, onset=0.0):
    """ Double Gamma HRF model.

    Parameters
    ----------
    delta : float, temporal dilation to pilot the HRF inflation
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    dur : float, (default=60.0), the time duration on which to represent the
        HRF
    onset : float, (default=0.0), onset of the HRF

    Return
    ------
    hrf : array, shape (dur / t_r, ), HRF
    """
    # dur: the (continious) time segment on which we represent all
    # the HRF. Can cut the HRF too early. The time is second.
    t = np.linspace(0, dur, int(float(dur) / DT)) - float(onset) / DT
    t = t[::int(t_r/DT)]

    _dilated_gamma_pdf_hrf_peak = _gamma_pdf_hrf_peak(delta * t)
    _dilated_gamma_pdf_hrf_undershoot = _gamma_pdf_hrf_undershoot(delta * t)

    c_1 = (P_DELAY - 1) * _dilated_gamma_pdf_hrf_peak
    c_2 = - P_DISP * delta * t * _dilated_gamma_pdf_hrf_peak
    c_3 = - (UNDERSHOOT - 1) * _dilated_gamma_pdf_hrf_undershoot
    c_4 = U_DISP * delta * t * _dilated_gamma_pdf_hrf_undershoot

    grad = (c_1 + c_2 + P_U_RATIO * (c_3 + c_4)) / delta

    return grad, t


# Double Gamma HRF


def double_gamma_hrf(t_r, n_times_atom=60):
    """ Double gamma HRF.

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
    _hrf = _double_gamma_hrf(delta=1.0, t_r=t_r, dur=n_times_atom * t_r)[0]
    return check_len_hrf(_hrf, n_times_atom)


def delta_derivative_double_gamma_hrf(delta, t_r, n_times_atom=60):
    """ Double gamma HRF derivative w.r.t the time dilation parameter.

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
    grad_ = _delta_derivative_double_gamma_hrf(delta=float(delta), t_r=t_r,
                                               dur=n_times_atom * t_r)[0]
    return check_len_hrf(grad_, n_times_atom)


# HRF models


def _hrf_d_basis(d, t_r, n_times_atom):
    """ Private helper to define the double gamma HRF 2/3 basis function.

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

    h_1 = _double_gamma_hrf(delta=1.0, t_r=t_r, dur=dur)[0]
    h_2_ = _double_gamma_hrf(delta=1.0, t_r=t_r, dur=dur, onset=0.0)[0]
    h_2__ = _double_gamma_hrf(delta=1.0, t_r=t_r, dur=dur, onset=t_r)[0]
    # h3 is derived w.r.t p_disp variable... can't used precomputed fonction
    t = np.linspace(0, dur, int(float(dur) / DT))
    peak = _gamma_pdf(t, P_DELAY/1.001, loc=DT/1.001)
    undershoot = _gamma_pdf_hrf_undershoot(t)
    h_3_ = (peak - P_U_RATIO * undershoot)[::int(t_r/DT)]

    h_2 = h_2_ - h_2__
    h_3 = (h_1 - h_3_) / 0.001

    h_1 = check_len_hrf(h_1, n_times_atom)
    h_2 = check_len_hrf(h_2, n_times_atom)
    h_3 = check_len_hrf(h_3, n_times_atom)

    if d == 2:
        return np.c_[h_1, h_2].T
    else:
        return np.c_[h_1, h_2, h_3].T


def hrf_2_basis(t_r, n_times_atom):
    """ Double gamma HRF 2 basis function.

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
    return _hrf_d_basis(2, t_r, n_times_atom)


def hrf_3_basis(t_r, n_times_atom):
    """ Double gamma HRF 3 basis function.

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
    return _hrf_d_basis(3, t_r, n_times_atom)


def scaled_hrf(delta, t_r, n_times_atom=60):
    """ Double gamma scaled HRF.

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
    _hrf = _double_gamma_hrf(delta=float(delta),
                             t_r=t_r, dur=n_times_atom * t_r)[0]
    return check_len_hrf(_hrf, n_times_atom)
