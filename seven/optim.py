"""Optimisation module"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np

from .loss_grad import _subsampled_cd_iter
from .prox import _prox_positive_L2_ball


def scdclinmodel(
        u0, r, constants=dict(), max_iter=100, early_stopping=False,
        eps=np.finfo(np.float64).eps, obj=None, benchmark=False):
    """ Subsampled coordinate descente on constraint linear model.
    """
    if benchmark and obj is None:
        raise ValueError("If 'benchmark' is set True 'obj' should be given.")

    n_atoms, n_channels = u0.shape
    u, u_old = np.copy(u0), np.copy(u0)

    if benchmark:
        pobj, times = [obj(u)], [0.0]

    C = constants['C']
    B = constants['B']
    step_size = 1.0 / np.diag(C)

    if (r != 1):
        block_len = int(n_channels / r)
        offsets = np.arange(n_channels)[::block_len]
        norm_u = np.sum(u0 * u0, axis=1)

    for ii in range(max_iter):

        if benchmark:
            t0 = time.process_time()

        if (r != 1):
            offset = offsets[ii % r]
            _subsampled_cd_iter(C, u, B, step_size, norm_u, offset, block_len)
        else:
            for k in range(n_atoms):
                grad_ = C[k, :].dot(u) - B[k, :]
                u[k, :] -= step_size[k] * grad_
                u[k, :] = _prox_positive_L2_ball(u[k, :])

        if benchmark:
            t1 = time.process_time()
            pobj.append(obj(u))

        if (r != 1):
            sub_u_old_view = u_old[:, offset: offset + block_len]
            diff = u[:, offset: offset + block_len] - sub_u_old_view
            norm_diff = diff.ravel().dot(diff.ravel())
            norm_u_old = sub_u_old_view.ravel().dot(sub_u_old_view.ravel())
        else:
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
    """ Proximal descent.
    """
    if benchmark and obj is None:
        raise ValueError("If 'benchmark' is set True 'obj' should be given.")

    if restarting == 'obj' and obj is None:
        raise ValueError("If 'restarting' is set 'obj' 'obj' should be given.")

    n_atoms, n_channels = x0.shape
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

        if restarting == 'obj' and (ii > 0) and (pobj[-1] > pobj[-2]):
            if momentum == 'fista':
                x = x_old
                t = 1.0
            elif momentum == 'greedy':
                y = x
        if restarting == 'descent':
            angle = (y_old - x).ravel().dot((x - x_old).ravel())
            if angle >= 0.0:
                if momentum == 'fista':
                    x = x_old
                    t = 1.0
                elif momentum == 'greedy':
                    y = x

        if benchmark:
            t1 = time.process_time()
            pobj.append(obj(y))

        converged = np.linalg.norm(x - x_old) < eps * np.linalg.norm(x_old)
        if early_stopping and converged:
            break

        t_old = t
        x_old = x
        y_old = y

        if benchmark:
            times.append(t1 - t0)

    if benchmark:
        return y, np.array(pobj), np.array(times)
    else:
        return y
