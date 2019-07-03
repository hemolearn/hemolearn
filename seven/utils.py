"""Utility module"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
from datetime import datetime

from alphacsc.utils.convolution import _dense_tr_conv_d


class EarlyStopping(Exception):
    """ Raised when the algorithm converged."""
    pass


def check_obj(lobj, ii, max_iter, early_stopping=True, raise_on_increase=True,
              eps=np.finfo(np.float64).eps):
    """ If raise_on_increase is True raise a RunTimeError exception when the
    objectif function has increased. Raise a EarlyStopping exception if the
    algorithm converged.
    """
    eps_Lz = ((lobj[-4] - lobj[-2]) / lobj[-4])
    eps_u = ((lobj[-3] - lobj[-1]) / lobj[-3])

    # check increasing cost-function
    if raise_on_increase and eps_Lz < eps:
        raise RuntimeError("[{}/{}] Updating Lz relatively increase "
                            "global cost-function of "
                            "{:.3e}".format(ii + 1, max_iter, -eps_Lz))
    if raise_on_increase and eps_u < eps:
        raise RuntimeError("[{}/{}] Updating u relatively increase "
                            "global cost-function of "
                            "{:.3e}".format(ii + 1, max_iter, -eps_u))

    # check early-stopping
    if early_stopping and eps_Lz < eps and eps_u < eps:
        msg = ("[{}/{}] Early-stopping done with: Lz-eps={:.3e}, "
               "u-eps={:.3e}".format(ii + 1, max_iter, eps_Lz, eps_u))
        raise EarlyStopping(msg)


def raise_if_vanished(A, msg="Vanished raw", eps=np.finfo(np.float64).eps):
    """ Raise an AssertionException if one raw of A has negligeable
    l2-norm.
    """
    norm_A_k = [A_k.dot(A_k) for A_k in A]
    check_A_k_nonzero = norm_A_k > eps
    assert np.all(check_A_k_nonzero), msg


def get_lambda_max(X, u, v_):
    """ Get lambda max.
    """
    n_atoms, n_channels = u.shape
    D_hat = np.c_[u, v_]
    DtX = _dense_tr_conv_d(X, D=D_hat, n_channels=n_channels)[:, None]
    return np.max(np.abs(DtX))


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


def sort_atoms_by_explained_variances(u_hat, Lz_hat, v=None):
    """ Sorted the temporal the spatial maps and the associated activation by
    explained variance."""
    assert Lz_hat.shape[0] == u_hat.shape[0]
    n_atoms, _ = u_hat.shape
    variances = np.zeros(n_atoms)
    if v is not None:
        vLz = np.r_[[np.convolve(v, Lz_hat_k) for Lz_hat_k in Lz_hat]]
    else:
        vLz = Lz_hat
    for k in range(n_atoms):
        variances[k] = np.var(np.outer(u_hat[k, :], vLz[k, :]))
    order = np.argsort(variances)[::-1]
    return u_hat[order, :], Lz_hat[order, :], variances[order]


def get_unique_dirname(prefix):
    """ Return a unique dirname based on the time and the date."""
    msg = "prefix should be a string, got {}".format(prefix)
    assert isinstance(prefix, str), msg
    date = datetime.now()
    date_tag = '{0}{1:02d}{2:02d}{3:02d}{4:02d}{5:02d}'.format(
                                        date.year, date.month, date.day,
                                        date.hour, date.minute, date.second)
    return prefix + date_tag


def check_lbda(lbda, lbda_strategy, X, u_hat, v_):
    """ Return the regularization factor."""
    if lbda_strategy not in ['ratio', 'fixed']:
        raise ValueError("'lbda_strategy' should belong to "
                         "['ratio', 'fixed'], got '{}'".format(lbda_strategy))

    if lbda_strategy == 'ratio':
        lbda_max = get_lambda_max(X, u_hat, v_)
        lbda = lbda * lbda_max
    else:
        if not isintance(lbda, (int, float)):
            raise ValueError("If 'lbda_strategy' is 'fixed', 'lbda' should be "
                             "numerical, got '{}'".format(lbda_strategy))
    return lbda
