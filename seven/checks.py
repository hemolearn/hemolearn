"""Atlas fetcher. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
from .convolution import adjconv_uH


class EarlyStopping(Exception):
    """ Raised when the algorithm converged."""


def check_obj(lobj, ii, max_iter, early_stopping=True, raise_on_increase=True,
              eps=np.finfo(np.float64).eps, level=1):
    """ If raise_on_increase is True raise a RunTimeError exception when the
    objectif function has increased. Raise a EarlyStopping exception if the
    algorithm converged.
    """
    if level == 1:
        _check_obj_level_1(lobj, ii, max_iter, early_stopping=early_stopping,
                           raise_on_increase=raise_on_increase, eps=eps)

    if level == 2:
        _check_obj_level_2(lobj, ii, max_iter, early_stopping=early_stopping,
                           raise_on_increase=raise_on_increase, eps=eps)


def _check_obj_level_1(lobj, ii, max_iter, early_stopping=True,
                       raise_on_increase=True, eps=np.finfo(np.float64).eps):
    """ Check after each iteration. """
    eps_ = (lobj[-2] - lobj[-1]) / lobj[-2]

    # check increasing cost-function
    if raise_on_increase and eps_ < eps:
        raise RuntimeError("[{}/{}] Iteration relatively increase "
                           "global cost-function of "
                           "{:.3e}".format(ii + 1, max_iter, -eps_))

    # check early-stopping
    if early_stopping and eps_ <= eps:
        msg = ("[{}/{}] Early-stopping (!) with: "
               "eps={:.3e}".format(ii + 1, max_iter, eps_))
        raise EarlyStopping(msg)


def _check_obj_level_2(lobj, ii, max_iter, early_stopping=True,
                       raise_on_increase=True, eps=np.finfo(np.float64).eps):
    """ Check after each update. """
    eps_z = (lobj[-4] - lobj[-3]) / lobj[-4]
    eps_u = (lobj[-3] - lobj[-2]) / lobj[-3]
    eps_v = (lobj[-2] - lobj[-1]) / lobj[-2]

    # check increasing cost-function
    if raise_on_increase and eps_z < eps:
        raise RuntimeError("[{}/{}] Updating z relatively increase "
                           "global cost-function of "
                           "{:.3e}".format(ii + 1, max_iter, -eps_z))
    if raise_on_increase and eps_u < eps:
        raise RuntimeError("[{}/{}] Updating u relatively increase "
                           "global cost-function of "
                           "{:.3e}".format(ii + 1, max_iter, -eps_u))
    if raise_on_increase and eps_v < eps:
        raise RuntimeError("[{}/{}] Updating v relatively increase "
                           "global cost-function of "
                           "{:.3e}".format(ii + 1, max_iter, -eps_v))

    # check early-stopping
    if early_stopping and eps_z <= eps and eps_u <= eps and eps_v <= eps:
        msg = ("[{}/{}] Early-stopping (!) with: z-eps={:.3e}, "
               "u-eps={:.3e}, v-eps={:.3e}".format(ii + 1, max_iter, eps_z,
                                                   eps_u, eps_v))
        raise EarlyStopping(msg)


def check_len_hrf(h, n_times_atom):
    """ Check thath the HRF has the proper length. """
    n = n_times_atom - len(h)
    if n < 0:
        h = h[:n]
    elif n > 0:
        h = np.hstack([h, np.zeros(n)])
    return h


def check_if_vanished(A, msg="Vanished raw", eps=np.finfo(np.float64).eps):
    """ Raise an AssertionException if one raw of A has negligeable
    l2-norm.
    """
    norm_A_k = [A_k.dot(A_k) for A_k in A]
    check_A_k_nonzero = norm_A_k > eps
    assert np.all(check_A_k_nonzero), msg


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance. """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{0} cannot be used to seed a numpy.random.RandomState'
                     ' instance'.format(seed))


def _get_lambda_max(X, u, H, rois_idx):
    """ Get lambda max. """
    uvtX = adjconv_uH(residual=X, u=u, H=H, rois_idx=rois_idx)[:, None]
    return np.max(np.abs(uvtX))


def check_lbda(lbda, lbda_strategy, X, u, H, rois_idx):
    """ Return the regularization factor."""
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
