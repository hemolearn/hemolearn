"""Main decomposition function"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import time
import numpy as np
from threadpoolctl import threadpool_limits

from alphacsc.utils.convolution import _dense_tr_conv_d
from alphacsc.utils.compute_constants import _compute_DtD_uv

from .utils import (lipschitz_est, check_random_state, get_lambda_max,
                    check_lbda, raise_if_vanished, check_obj, EarlyStopping)
from .optim import scdclinmodel, proximal_descent
from .loss_grad import _grad_Lz, obj_u_z
from .prox import _prox_tv_multi


def _update_u(u0, constants):
    """ Update spatiel maps."""
    with threadpool_limits(limits=1, user_api='blas'):
        msg = "'C' is missing in 'constants' for '_update_u' step."
        assert ('C' in constants), msg
        msg = "'B' is missing in 'constants' for '_update_u' step."
        assert ('B' in constants), msg

        params = dict(u0=u0, r=1, max_iter=100, constants=constants)
        u_hat = scdclinmodel(**params)

    return u_hat


def _update_Lz(Lz0, constants):
    """ Update temporal components."""
    msg = "'uv' is missing in 'constants' for '_update_Lz' step."
    assert ('uv' in constants), msg
    msg = "'X' is missing in 'constants' for '_update_Lz' step."
    assert ('X' in constants), msg
    msg = "'lbda' is missing in 'constants' for '_update_Lz' step."
    assert ('lbda' in constants), msg

    uv = constants['uv']
    X = constants['X']
    lbda = constants['lbda']

    n_channels, _ = X.shape
    DtD = _compute_DtD_uv(uv, n_channels)
    DtX = _dense_tr_conv_d(X, D=uv, n_channels=n_channels)

    def prox(Lz, step_size):
        return _prox_tv_multi(Lz, lbda, step_size)

    def grad(Lz):
        return _grad_Lz(Lz, DtD, DtX)

    def AtA(Lz):
        return _grad_Lz(Lz, DtD)
    step_size = 0.9 / lipschitz_est(AtA, Lz0.shape)

    params = dict(x0=Lz0, grad=grad, prox=prox, step_size=step_size,
                  momentum='fista', restarting='descent', max_iter=100)
    Lz_hat = proximal_descent(**params)

    return Lz_hat


def lean_u_z_multi(X, v, n_atoms, lbda_strategy='ratio', lbda=0.1,
                   max_iter=50, get_obj=False, get_time=False,
                   u_init='random', random_state=None, name="DL",
                   early_stopping=True, eps=1.0e-5, raise_on_increase=True,
                   verbose=False):
    """ Multivariate Convolutional Sparse Coding with n_atoms-rank constraint.
    """
    X = X.astype(np.float64)
    v_ = np.repeat(v[None, :], n_atoms, axis=0).squeeze()

    rng = check_random_state(random_state)

    n_channels, n_times = X.shape
    n_times_atom = v.shape[0]
    n_times_valid = n_times - n_times_atom + 1

    Lz_hat = np.zeros((n_atoms, n_times_valid))
    if u_init == 'random':
        u_hat = rng.randn(n_atoms, n_channels)
    else:
        raise ValueError("u_init should be in ['random', ]"
                         ", got {}".format(u_init))

    if (raise_on_increase or early_stopping) and not get_obj:
        raise ValueError("raise_on_increase or early_stopping can only be set"
                         "to True if get_obj is True")

    lbda = check_lbda(lbda, lbda_strategy, X, u_hat, v_)

    constants = dict(lbda=lbda, X=X)
    if get_obj:
        lobj = [obj_u_z(X=X, lbda=lbda, u=u_hat, v=v, Lz=Lz_hat, valid=True)]
    if get_time:
        ltime = [0.0]

    for ii in range(max_iter):

        # Update Lz
        constants['uv'] = np.c_[u_hat, v_]

        if get_time:
            t0 = time.process_time()
        Lz_hat = _update_Lz(Lz_hat, constants)  # update
        if get_time:
            ltime.append(time.process_time() - t0)

        A = np.r_[[np.convolve(v, Lz_hat_k) for Lz_hat_k in Lz_hat]].T
        constants['C'] = A.T.dot(A)
        constants['B'] = A.T.dot(X.T)

        if get_obj:
            obj_ = obj_u_z(X=X, lbda=lbda, u=u_hat, Lz=Lz_hat, v=v, valid=True)
            lobj.append(obj_)
            if verbose > 1:
                if get_time:
                    print("[{}/{}] Update Lz done: {:.4f} in {:.3f}s".format(
                                ii + 1, max_iter, obj_ / lobj[0], ltime[-1]))
                else:
                    print("[{}/{}] Update Lz done: {:.4f}".format(
                                            ii + 1, max_iter, obj_ / lobj[0]))

        # check if some Lz_k vanished
        msg = ("Temporal component vanished, may be 'lbda' is too high, "
               "please try to reduce its value.")
        raise_if_vanished(Lz_hat, msg)

        # Update u
        if get_time:
            t0 = time.process_time()
        u_hat = _update_u(u_hat, constants)  # update
        if get_time:
            ltime.append(time.process_time() - t0)

        if get_obj:
            obj_ = obj_u_z(X=X, lbda=lbda, u=u_hat, Lz=Lz_hat, v=v, valid=True)
            lobj.append(obj_)
            if verbose > 1:
                if get_time:
                    print("[{}/{}] Update u done: {:.4f} in {:.3f}s".format(
                                ii + 1, max_iter, obj_ / lobj[0], ltime[-1]))
                else:
                    print("[{}/{}] Update u done: {:.4f}".format(
                                            ii + 1, max_iter, obj_ / lobj[0]))

        if ii > 2 and get_obj:
            try:
                check_obj(lobj, raise_on_increase, eps)
            except EarlyStopping as e:
                if verbose > 1:
                    print(str(e))
                break

    z_hat = np.diff(Lz_hat, axis=-1)

    if get_obj and get_time:
        return Lz_hat, z_hat, u_hat, lbda, np.array(lobj), np.array(ltime)
    elif get_obj and not get_time:
        return Lz_hat, z_hat, u_hat, lbda, np.array(lobj), None
    elif not get_obj and get_time:
        return Lz_hat, z_hat, u_hat, lbda, None, np.array(ltime)
    else:
        return Lz_hat, z_hat, u_hat, lbda, None, None
