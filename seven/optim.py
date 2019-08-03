"""Optimisation module"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import time
import numpy as np

from .loss_grad import _grad_u_k
from .prox import _prox_positive_L2_ball


def cdclinmodel(
        u0, constants=dict(), max_iter=100, early_stopping=False,
        eps=np.finfo(np.float64).eps, obj=None, benchmark=False):
    """ Coordinate descente on constraint linear model:
    grad_u(u) = C.dot(u) - B and prox_u(.) = _prox_positive_L2_ball(.)
    (see prox.py for the definition of _prox_positive_L2_ball)

    Parameters
    ----------
    u0 : array, shape (n_atoms, n_voxels), initial spatial maps
    constants : dict, gather the usefull constant for the estimation of the
        spatial maps, keys are (C, B)
    max_iter : int, (default=100), maximum number of iterations to perform the
        algorithm
    early_stopping : bool, (default=True), whether to early stop the analysis
    eps : float, (default=np.finfo(np.float64).eps), stoppping parameter w.r.t
        evolution of the cost-function
    obj : func, (default=None), cost-function function
    benchmark : bool, (default=False), whether or not to save the cost-function
        and the duration of computatio nof each iteration

    Return
    ------
    u : array, shape (n_atoms, n_voxels), the estimated spatial maps
    pobj : array or None, shape (n_iter,) or (3 * n_iter,), the saved
        cost-function
    times : array or None, shape (n_iter,) or(3 * n_iter,), the saved duration
        per steps
    """
    if benchmark and obj is None:
        raise ValueError("If 'benchmark' is set True 'obj' should be given.")

    n_atoms, n_voxels = u0.shape
    u, u_old = np.copy(u0), np.copy(u0)

    if benchmark:
        pobj, times = [obj(u)], [0.0]

    C = constants['C']
    B = constants['B']
    rois_idx = constants['rois_idx']
    n_hrf_rois = len(rois_idx)

    diag = [np.diag(C[m, :, :]) for m in range(n_hrf_rois)]
    step_size = 1.0 / np.r_[diag].max(axis=0)

    for _ in range(max_iter):

        if benchmark:
            t0 = time.process_time()

        for k in range(n_atoms):
            u[k, :] -= step_size[k] * _grad_u_k(u, B, C, k, rois_idx)
            u[k, :] = _prox_positive_L2_ball(u[k, :])

        if benchmark:
            t1 = time.process_time()
            pobj.append(obj(u))

        diff = u - u_old
        norm_diff = diff.ravel().dot(diff.ravel())
        norm_u_old = u_old.ravel().dot(u_old.ravel())
        if early_stopping and norm_diff < eps * norm_u_old:
            break

        u_old = u

        if benchmark:
            times.append(t1 - t0)

    if benchmark:
        return u, np.array(pobj), np.array(times)
    else:
        return u


def proximal_descent(
            x0, grad, prox, step_size, momentum='fista', restarting=None,
            max_iter=100, early_stopping=True, eps=np.finfo(np.float64).eps,
            obj=None, benchmark=False):
    """ Proximal descent algorithm.

    Parameters
    ----------
    x0 : array, shape (n_length, ), initial variables
    grad : func, gradient function
    prox : func, proximal operator function
    step_size : float, step-size for the gradient descent
    momentum : str or None, (default='fista'), momentum to choose, possible
        choice are ('fista', 'greedy', None)
    restarting : str or None, (default=None), restarting to chosse, possible
        choice are ('obj', 'descent', None),  if restarting == 'obj', obj
        function should be given
    max_iter : int, (default=100), maximum number of iterations to perform the
        analysis
    early_stopping : bool, (default=True), whether to early stop the analysis
    eps : float, (default=np.finfo(np.float64).eps), stoppping parameter w.r.t
        evolution of the cost-function
    obj : func, (default=None), cost-function function
    benchmark : bool, (default=False), whether or not to save the cost-function
        and the duration of computatio nof each iteration

    Return
    ------
    x : array, shape (n_atoms, n_voxels), the estimated variable
    pobj : array or None, shape (n_iter,) or (3 * n_iter,), the saved
        cost-function
    times : array or None, shape (n_iter,) or(3 * n_iter,), the saved duration
        per steps
    """
    if benchmark and obj is None:
        raise ValueError("If 'benchmark' is set True 'obj' should be given.")

    if restarting == 'obj' and obj is None:
        raise ValueError("If 'restarting' is set 'obj' 'obj' should be given.")

    x_old, x, y, y_old = np.copy(x0), np.copy(x0), np.copy(x0), np.copy(x0)
    t = t_old = 1
    if benchmark:
        pobj, times = [obj(y)], [0.0]

    for ii in range(max_iter):

        if benchmark:
            t0 = time.process_time()

        y -= step_size * grad(y)
        x = prox(y, step_size)

        if momentum == 'fista':
            t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_old**2))
            y = x + (t_old - 1.0) / t * (x - x_old)
        elif momentum == 'greedy':
            y = x + (x - x_old)
        elif momentum is None:
            y = x

        restarted = False
        if restarting == 'obj' and (ii > 0) and (pobj[-1] > pobj[-2]):
            if momentum == 'fista':
                x = x_old
                t = 1.0
            elif momentum == 'greedy':
                y = x
            restarted = True
        if restarting == 'descent':
            angle = (y_old - x).ravel().dot((x - x_old).ravel())
            if angle >= 0.0:
                if momentum == 'fista':
                    x = x_old
                    t = 1.0
                elif momentum == 'greedy':
                    y = x
                restarted = True

        if benchmark:
            t1 = time.process_time()
            pobj.append(obj(y))

        converged = np.linalg.norm(x - x_old) < eps * np.linalg.norm(x_old)
        if early_stopping and converged and not restarted:
            break

        t_old = t
        x_old = x
        y_old = y

        if benchmark:
            times.append(t1 - t0)

    if benchmark:
        return x, np.array(pobj), np.array(times)
    else:
        return x
