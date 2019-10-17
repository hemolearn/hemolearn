"""Checks module: gather utilities to perform routine checks. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
from .convolution import adjconv_uH


class EarlyStopping(Exception):
    """ Raised when the algorithm converged."""


class CostFunctionIncreased(Exception):
    """ Raised when the cost-function has increased."""


def check_obj(lobj, ii, max_iter, early_stopping=True, raise_on_increase=True,
              eps=np.finfo(np.float64).eps, level=1):
    """ If raise_on_increase is True raise a CostFunctionIncreased exception
    when the objectif function has increased. Raise a EarlyStopping exception
    if the algorithm converged.

    Parameters
    ----------
    lobj : array or None, shape (n_iter,) or (3 * n_iter,), the saved
        cost-function
    ii : int, the index of the current iteration
    max_iter : int, (default=100), maximum number of iterations to perform the
        analysis
    early_stopping : bool, (default=True), whether to early stop the analysis
    raise_on_increase : bool, (default=True), whether to stop the analysis if
        the cost-function increases during an iteration. This can be due to the
        fact that the temporal regularization parameter is set to high
    eps : float, (default=np.finfo(np.float64).eps, stoppping parameter w.r.t
        evolution of the cost-function
    level : int, (default=1), desired level of cost-function monitoring, 1 for
        cost-function computation at each iteration, 2 for cost-function
        computation at each steps (z, u, v)

    Throws
    ------
    CostFunctionIncreased : if the cost-function increases during an iteration,
        of the analysis. This can be due to the fact that the temporal
        regularization parameter is set to high
    EarlyStopping : if the cost-function has converged
    """
    if level == 1:
        _check_obj_level_1(lobj, ii, max_iter, early_stopping=early_stopping,
                           raise_on_increase=raise_on_increase, eps=eps)

    if level == 2:
        _check_obj_level_2(lobj, ii, max_iter, early_stopping=early_stopping,
                           raise_on_increase=raise_on_increase, eps=eps)


def _check_obj_level_1(lobj, ii, max_iter, early_stopping=True,
                       raise_on_increase=True, eps=np.finfo(np.float64).eps):
    """ Check after each iteration.

    Parameters
    ----------
    lobj : array or None, shape (n_iter,) or (3 * n_iter,), the saved
        cost-function
    ii : int, the index of the current iteration
    max_iter : int, (default=100), maximum number of iterations to perform the
        analysis
    early_stopping : bool, (default=True), whether to early stop the analysis
    raise_on_increase : bool, (default=True), whether to stop the analysis if
        the cost-function increases during an iteration. This can be due to the
        fact that the temporal regularization parameter is set to high
    eps : float, (default=np.finfo(np.float64).eps, stoppping parameter w.r.t
        evolution of the cost-function

    Throws
    ------
    CostFunctionIncreased : if the cost-function increases during an iteration,
        of the analysis. This can be due to the fact that the temporal
        regularization parameter is set to high
    EarlyStopping : if the cost-function has converged
    """
    eps_ = (lobj[-2] - lobj[-1]) / lobj[-2]

    # check increasing cost-function
    if raise_on_increase and eps_ < eps:
        raise CostFunctionIncreased(
                           "[{}/{}] Iteration relatively increase "
                           "global cost-function of "
                           "{:.3e}".format(ii, max_iter, -eps_))

    # check early-stopping
    if early_stopping and np.abs(eps_) <= eps:
        msg = ("[{}/{}] Early-stopping (!) with: "
               "eps={:.3e}".format(ii, max_iter, eps_))
        raise EarlyStopping(msg)


def _check_obj_level_2(lobj, ii, max_iter, early_stopping=True,
                       raise_on_increase=True, eps=np.finfo(np.float64).eps):
    """ Check after each update.

    Parameters
    ----------
    lobj : array or None, shape (n_iter,) or (3 * n_iter,), the saved
        cost-function
    ii : int, the index of the current iteration
    max_iter : int, (default=100), maximum number of iterations to perform the
        analysis
    early_stopping : bool, (default=True), whether to early stop the analysis
    raise_on_increase : bool, (default=True), whether to stop the analysis if
        the cost-function increases during an iteration. This can be due to the
        fact that the temporal regularization parameter is set to high
    eps : float, (default=np.finfo(np.float64).eps, stoppping parameter w.r.t
        evolution of the cost-function

    Throws
    ------
    CostFunctionIncreased : if the cost-function increases during an iteration,
        of the analysis. This can be due to the fact that the temporal
        regularization parameter is set to high
    EarlyStopping : if the cost-function has converged
    """
    eps_z = (lobj[-4] - lobj[-3]) / lobj[-4]
    eps_u = (lobj[-3] - lobj[-2]) / lobj[-3]
    eps_v = (lobj[-2] - lobj[-1]) / lobj[-2]

    # check increasing cost-function
    if raise_on_increase and eps_z < eps:
        raise CostFunctionIncreased(
                           "[{}/{}] Updating z relatively increase "
                           "global cost-function of "
                           "{:.3e}".format(ii + 1, max_iter, -eps_z))
    if raise_on_increase and eps_u < eps:
        raise CostFunctionIncreased(
                           "[{}/{}] Updating u relatively increase "
                           "global cost-function of "
                           "{:.3e}".format(ii + 1, max_iter, -eps_u))
    if raise_on_increase and eps_v < eps:
        raise CostFunctionIncreased(
                           "[{}/{}] Updating v relatively increase "
                           "global cost-function of "
                           "{:.3e}".format(ii + 1, max_iter, -eps_v))

    # check early-stopping
    eps_check = (np.abs(eps_z) <= eps)
    eps_check = eps_check and (np.abs(eps_u) <= eps)
    eps_check = eps_check and (np.abs(eps_v) <= eps)
    if (early_stopping and eps_check):
        msg = ("[{}/{}] Early-stopping (!) with: z-eps={:.3e}, "
               "u-eps={:.3e}, v-eps={:.3e}".format(ii + 1, max_iter, eps_z,
                                                   eps_u, eps_v))
        raise EarlyStopping(msg)


def check_len_hrf(h, n_times_atom):
    """ Check that the HRF has the proper length.

    Parameters
    ----------
    h : array, shape (n_times_atom, ), HRF
    n_times_atom : int, number of components on which to decompose the neural
        activity (number of temporal components and its associated spatial
        maps).

    Return
    ------
    h : array, shape (n_times_atom, ), HRF with a correct length
    """
    n = n_times_atom - len(h)
    if n < 0:
        h = h[:n]
    elif n > 0:
        h = np.hstack([h, np.zeros(n)])
    return h


def check_if_vanished(A, msg="Vanished raw", eps=np.finfo(np.float64).eps):
    """ Raise an AssertionException if one raw of A has negligeable
    l2-norm.

    Parameters
    ----------
    A : array, the array on which to check if a raw is too close to zero
    msg : str, (default="Vanished raw"), message to display with the
        AssertionError exception
    eps : float, (default=np.finfo(np.float64).eps), tolerance among the
        squared l2-norm of the raws

    Throws
    ------
        AssertionError : if a raw has vanished
    """
    norm_A_k = [A_k.dot(A_k) for A_k in A]
    check_A_k_nonzero = norm_A_k > eps
    assert np.all(check_A_k_nonzero), msg


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance

    Return
    ------
    random_instance : random-instance used to initialize the analysis
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{0} cannot be used to seed a numpy.random.RandomState'
                     ' instance'.format(seed))


def _get_lambda_max(X, u, H, rois_idx):
    """ Get lbda max (the maximum value of the temporal regularization
    parameter which systematically provide zero temporal components).

    Parameters
    ----------
    X : array, shape (n_voxels, n_times), fMRI data
    u : array, shape (n_atoms, n_voxels), spatial maps
    H : array, shape (n_hrf_rois, n_times_valid, n_times), Toeplitz matrices
    rois_idx: array, shape (n_hrf_rois, max_indices_per_rois), HRF ROIs

    Return
    ------
    lbda_max : float, the maximum value of the temporal regularization
        parameter which systematically provide zero temporal components
    """
    uvtX = adjconv_uH(residual=X, u=u, H=H, rois_idx=rois_idx)[:, None]
    return np.max(np.abs(uvtX))


def check_lbda(lbda, lbda_strategy, X, u, H, rois_idx):
    """ Return the temporal regularization parameter.

    Parameters
    ----------
    lbda : float, (default=0.1), whether the temporal regularization parameter
        if lbda_strategy == 'fixed' or the ratio w.r.t lambda max if
        lbda_strategy == 'ratio'
    lbda_strategy str, (default='ratio'), strategy to fix the temporal
        regularization parameter, possible choice are ['ratio', 'fixed']
    X : array, shape (n_voxels, n_times), fMRI data
    u : array, shape (n_atoms, n_voxels), spatial maps
    H : array, shape (n_hrf_rois, n_times_valid, n_times), Toeplitz matrices
    rois_idx: array, shape (n_hrf_rois, max_indices_per_rois), HRF ROIs

    Return
    ------
    lbda : float, the value of the temporal regularization parameter
    """
    if lbda_strategy not in ['ratio', 'fixed']:
        raise ValueError("'lbda_strategy' should belong to "
                         "['ratio', 'fixed'], got '{}'".format(lbda_strategy))
    if lbda_strategy == 'ratio':
        lbda_max = _get_lambda_max(X, u=u, H=H, rois_idx=rois_idx)
        lbda = lbda * lbda_max
    else:
        if not isinstance(lbda, (int, float)):
            raise ValueError("If 'lbda_strategy' is 'fixed', 'lbda' should be "
                             "numerical, got '{}'".format(lbda_strategy))
    return lbda
