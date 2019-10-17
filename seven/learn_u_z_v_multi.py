"""Main module: main decomposition function"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import gc
import time
import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from sklearn.decomposition import FastICA

from .checks import (check_lbda, check_if_vanished, check_obj,
                     EarlyStopping, check_random_state)
from .utils import lipschitz_est
from .constants import _precompute_uvtuv, _precompute_B_C
from .optim import cdclinmodel, proximal_descent
from .loss_grad import _grad_z, _obj
from .prox import _prox_tv_multi
from .hrf_model import scaled_hrf, hrf_3_basis, hrf_2_basis
from .estim_v import _estim_v_d_basis, _estim_v_scaled_hrf
from .atlas import split_atlas
from .convolution import adjconv_uH, make_toeplitz
from .prox import _prox_l1_simplex, _prox_positive_l2_ball


def _update_v(a0, constants):
    """ Update the HRFs.

    Parameters
    ----------
    a0 : array, shape (n_hrf_rois, n_param_HRF), initial HRF parameters
    constants : dict, gather the usefull constant for the estimation of the
        HRF, keys are (z, u, rois_idx, X, hrf_model)

    Return
    ------
    a : array, shape (n_hrf_rois, n_param_HRF),  estimated HRF parameters
    v : array, shape (n_hrf_rois, n_times_atom), estimated HRFs
    """
    msg = "'z' is missing in 'constants' for '_update_v' step."
    assert ('z' in constants), msg
    msg = "'u' is missing in 'constants' for '_update_v' step."
    assert ('u' in constants), msg
    msg = "'rois_idx' is missing in 'constants' for '_update_z' step."
    assert ('rois_idx' in constants), msg
    msg = "'X' is missing in 'constants' for '_update_v' step."
    assert ('X' in constants), msg
    msg = "'hrf_model' is missing in 'constants' for '_update_v' step."
    assert ('hrf_model' in constants), msg

    z, u = constants['z'], constants['u']
    X, rois_idx = constants['X'], constants['rois_idx']
    hrf_model = constants['hrf_model']

    if hrf_model in ['2_basis_hrf', '3_basis_hrf']:
        msg = "'h' is missing in 'constants' for '_update_v' step."
        assert ('h' in constants), msg
        h = constants['h']
        return _estim_v_d_basis(a0, X, h, z, u, rois_idx)

    elif hrf_model == 'scaled_hrf':
        msg = "'t_r' is missing in 'constants' for '_update_v' step."
        assert ('t_r' in constants), msg
        msg = ("'n_times_atom' is missing in 'constants' for "
               "'_update_v' step.")
        assert ('n_times_atom' in constants), msg
        t_r, n_times_atom = constants['t_r'], constants['n_times_atom']
        return _estim_v_scaled_hrf(a0, X, z, u, rois_idx, t_r, n_times_atom)

    else:
        raise ValueError("hrf_model should be in ['3_basis_hrf', "
                         "'2_basis_hrf', 'scaled_hrf', 'fir_hrf'], "
                         "got {}".format(hrf_model))


def _update_u(u0, constants):
    """ Update the spatial maps.

    Parameters
    ----------
    u0 : array, shape (n_atoms, n_voxels), initial spatial maps
    constants : dict, gather the usefull constant for the estimation of the
        spatial maps, keys are (C, B)

    Return
    ------
    u : array, shape (n_atoms, n_voxels), estimated spatial maps
    """
    msg = "'C' is missing in 'constants' for '_update_u' step."
    assert ('C' in constants), msg
    msg = "'B' is missing in 'constants' for '_update_u' step."
    assert ('B' in constants), msg
    msg = "'prox_u' is missing in 'constants' for '_update_u' step."
    assert ('prox_u' in constants), msg

    params = dict(u0=u0, max_iter=100, constants=constants)
    u_hat = cdclinmodel(**params)

    return u_hat


def _update_z(z0, constants):
    """ Update the temporal components.

    Parameters
    ----------
    z0 : array, shape (n_atoms, n_voxels), initial temporal components
    constants : dict, gather the usefull constant for the estimation of the
        temporal components, keys are (v, H, u, rois_idx, X, lbda)

    Return
    ------
    z : array, shape (n_atoms, n_voxels), estimated temporal components
    """
    msg = "'v' is missing in 'constants' for '_update_z' step."
    assert ('v' in constants), msg
    msg = "'H' is missing in 'constants' for '_update_z' step."
    assert ('H' in constants), msg
    msg = "'u' is missing in 'constants' for '_update_z' step."
    assert ('u' in constants), msg
    msg = "'rois_idx' is missing in 'constants' for '_update_z' step."
    assert ('rois_idx' in constants), msg
    msg = "'X' is missing in 'constants' for '_update_z' step."
    assert ('X' in constants), msg
    msg = "'lbda' is missing in 'constants' for '_update_z' step."
    assert ('lbda' in constants), msg

    u = constants['u']
    H = constants['H']
    v = constants['v']
    rois_idx = constants['rois_idx']
    X = constants['X']
    lbda = constants['lbda']

    uvtuv = _precompute_uvtuv(u=u, v=v, rois_idx=rois_idx)
    uvtX = adjconv_uH(X, u=u, H=H, rois_idx=rois_idx)

    def prox(z, step_size):
        return _prox_tv_multi(z, lbda, step_size)

    def grad(z):
        return _grad_z(z, uvtuv, uvtX)

    def AtA(z):
        return _grad_z(z, uvtuv)
    step_size = 0.9 / lipschitz_est(AtA, z0.shape)

    params = dict(x0=z0, grad=grad, prox=prox, step_size=step_size,
                  momentum='fista', restarting='descent', max_iter=100)
    z_hat = proximal_descent(**params)

    return z_hat


def learn_u_z_v_multi(
        X, t_r, hrf_rois, hrf_model='scaled_hrf', deactivate_v_learning=False,
        n_atoms=10, n_times_atom=60, lbda_strategy='ratio', lbda=0.1,
        u_init_type='ica', prox_u='l1-positive-simplex', max_iter=100,
        get_obj=0, get_time=0, random_seed=None, early_stopping=True,
        eps=1.0e-5, raise_on_increase=True, verbose=0):
    """ Multivariate Convolutional Sparse Coding with n_atoms-rank constraint.

    Parameters
    ----------
    X : array, shape (n_voxels, n_times), fMRI data
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    hrf_rois : dict (key: ROIs labels, value: indices of voxels of the ROI)
        atlas HRF
    hrf_model : str, (default='3_basis_hrf'), type of HRF model, possible
        choice are ['3_basis_hrf', '2_basis_hrf', 'scaled_hrf']
    deactivate_v_learning : bool, (default=False), option to force the
        estimated HRF to to the initial value.
    n_atoms : int, number of components on which to decompose the neural
        activity (number of temporal components and its associated spatial
        maps).
    n_times_atom : int, (default=30), number of points on which represent the
        Haemodynamic Response Function (HRF), this leads to the duration of the
        response function, duration = n_times_atom * t_r
    lbda_strategy str, (default='ratio'), strategy to fix the temporal
        regularization parameter, possible choice are ['ratio', 'fixed']
    lbda : float, (default=0.1), whether the temporal regularization parameter
        if lbda_strategy == 'fixed' or the ratio w.r.t lambda max if
        lbda_strategy == 'ratio'
    u_init_type : str, (default='ica'), strategy to init u, possible value are
        ['gaussian_noise', 'ica', 'patch']
    prox_u : str, (default='l2-positive-ball'), constraint to impose on the
        spatial maps possible choice are ['l2-positive-ball',
        'l1-positive-simplex']
    max_iter : int, (default=100), maximum number of iterations to perform the
        analysis
    get_obj : int, the level of cost-function saving, 0 to not compute it, 1 to
        compute it at each iteration, 2 to compute it at each estimation step
        (z, u, v)
    get_time : int, the level of computation-duration saving 0 to not compute
        it, 1 to compute it at each iteration, 2 to compute it at each
        estimation step
    random_seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the analysis
    early_stopping : bool, (default=True), whether to early stop the analysis
    eps : float, (default=1.0e-4), stoppping parameter w.r.t evolution of the
        cost-function
    raise_on_increase : bool, (default=True), whether to stop the analysis if
        the cost-function increases during an iteration. This can be due to the
        fact that the temporal regularization parameter is set to high
    verbose : int, (default=0), verbose level, 0 no verbose, 1 low verbose,
        2 maximum verbose

    Return
    ------
    z : array, shape (n_atoms, n_times_valid), the estimated temporal
        components
    Dz : array, shape (n_atoms, n_times_valid), the estimated first order
        derivation temporal components
    u : array, shape (n_atoms, n_voxels), the estimated spatial maps
    a : array, shape (n_hrf_rois, n_param_HRF), the estimated HRF parameters
    v : array, shape (n_hrf_rois, n_times_atom), the estimated HRFs
    v : array, shape (n_hrf_rois, n_times_atom), the initial used HRFs
    lbda : float, the temporal regularization parameter used
    lobj : array or None, shape (n_iter,) or (3 * n_iter,), the saved
        cost-function
    ltime : array or None, shape (n_iter,) or(3 * n_iter,), the saved duration
        per steps

    Throws
    ------
    CostFunctionIncreased : if the cost-function increases during an iteration,
        of the analysis. This can be due to the fact that the temporal
        regularization parameter is set to high
    """
    X = X.astype(np.float64)

    rng = check_random_state(random_seed)

    n_voxels, n_times = X.shape
    n_times_valid = n_times - n_times_atom + 1

    if n_times_valid < 2 * n_times_atom - 1:
        raise ValueError("'n_times_atom' is too hight w.r.t the duration of "
                         "the acquisition, please reduce it.")

    # split atlas
    rois_idx, _, n_hrf_rois = split_atlas(hrf_rois)

    constants = dict(X=X, rois_idx=rois_idx, hrf_model=hrf_model)

    # v initialization
    if hrf_model == '3_basis_hrf':
        h = hrf_3_basis(t_r, n_times_atom)
        a_hat = np.c_[[np.array([1.0, 0.0, 0.0]) for _ in range(n_hrf_rois)]]
        v_hat = np.c_[[a_.dot(h) for a_ in a_hat]]
        constants['h'] = h

    elif hrf_model == '2_basis_hrf':
        h = hrf_2_basis(t_r, n_times_atom)
        a_hat = np.c_[[np.array([1.0, 0.0]) for _ in range(n_hrf_rois)]]
        v_hat = np.c_[[a_.dot(h) for a_ in a_hat]]
        constants['h'] = h

    elif hrf_model == 'scaled_hrf':
        delta = 1.0
        a_hat = delta * np.ones(n_hrf_rois)
        v_ = scaled_hrf(delta=delta, t_r=t_r, n_times_atom=n_times_atom)
        v_hat = np.c_[[v_ for a_ in a_hat]]
        constants['t_r'] = t_r
        constants['n_times_atom'] = n_times_atom

    else:
        raise ValueError("hrf_model should be in ['3_basis_hrf', '2_basis_hrf'"
                         ", 'scaled_hrf', 'fir_hrf'], "
                         "got {}".format(hrf_model))
    v_init = v_hat[0, :]

    # H initialization
    H_hat = np.empty((n_hrf_rois, n_times, n_times_valid))
    for m in range(n_hrf_rois):
        H_hat[m, :, :] = make_toeplitz(v_hat[m], n_times_valid=n_times_valid)

    # z initialization
    z_hat = np.zeros((n_atoms, n_times_valid))

    # u initialization
    if u_init_type == 'gaussian_noise':
        u_hat = rng.randn(n_atoms, n_voxels)
        for k in range(n_atoms):
            u_hat[k, :] = _prox_l1_simplex(u_hat[k, :], eta=10.0)

    elif u_init_type == 'ica':
        ica = FastICA(n_components=n_atoms, algorithm='deflation', max_iter=50)
        ica.fit(X.T)
        u_hat = np.copy(ica.components_)
        del ica  # heavy object
        gc.collect()
        for k in range(n_atoms):
            u_hat[k, :] = _prox_l1_simplex(u_hat[k, :], eta=10.0)

    elif u_init_type == 'patch':
        u_hat = np.empty((n_atoms, n_voxels))
        for k in range(n_atoms):
            idx_start = rng.randint(0, n_times - n_times_atom)
            idx_stop = idx_start + n_times_atom
            u_k_init = X[:, idx_start:idx_stop].dot(v_hat[0, :])
            u_hat[k, :] = _prox_l1_simplex(u_k_init, eta=10.0)

    else:
        raise ValueError("u_init_type should be in ['gaussian_noise', "
                         "'gaussian_noise', 'patch'],"
                         " got {}".format(u_init_type))

    if (raise_on_increase or early_stopping) and not get_obj:
        raise ValueError("raise_on_increase or early_stopping can only be set"
                         " to True if get_obj is True")

    # temporal regularization parameter
    lbda = check_lbda(lbda, lbda_strategy, X, u_hat, H_hat, rois_idx)

    # spatial regularization parameter
    if prox_u == 'l2-positive-ball':
        def _prox(u_k):
            return _prox_positive_l2_ball(u_k, step_size=1.0)
        prox_u_func = _prox
    elif prox_u == 'l1-positive-simplex':
        def _prox(u_k):
            return _prox_l1_simplex(u_k, eta=10.0)
        prox_u_func = _prox
    else:
        raise ValueError("prox_u should be in ['l2-positive-ball', "
                         "'l1-positive-simplex'], got {}".format(prox_u))

    constants['lbda'] = lbda
    constants['prox_u'] = prox_u_func

    if get_obj:
        lobj = [_obj(X=X, prox=prox_u_func, lbda=lbda, u=u_hat, z=z_hat,
                     H=H_hat, rois_idx=rois_idx, valid=True)]
    if get_time:
        ltime = [0.0]

    # main loop
    with threadpool_limits(limits=1):

        for ii in range(max_iter):

            if get_time == 1:
                t0 = time.process_time()

            # use Toeplitz matrices for obj. func. computation (Numpy speed-up)
            for m in range(n_hrf_rois):
                H_hat[m, ...] = make_toeplitz(v_hat[m],
                                              n_times_valid=n_times_valid)

            # Update z
            constants['a'] = a_hat
            constants['v'] = v_hat
            constants['H'] = H_hat
            constants['u'] = u_hat

            if get_time == 2:
                t0 = time.process_time()
            z_hat = _update_z(z_hat, constants)  # update
            if get_time == 2:
                ltime.append(time.process_time() - t0)

            if get_obj == 2:
                lobj.append(_obj(X=X, prox=prox_u_func, lbda=lbda, u=u_hat,
                                 z=z_hat, H=H_hat, rois_idx=rois_idx,
                                 valid=True))
                if verbose > 1:
                    if get_time:
                        print("[{}/{}] Update z done in {:.1f} s : "
                              "cost = {:.6f}".format(ii + 1, max_iter,
                                                     ltime[-1],
                                                     lobj[-1] / lobj[0]))
                    else:
                        print("[{}/{}] Update z done: "
                              "cost = {:.6f}".format(ii + 1, max_iter,
                                                     lobj[-1] / lobj[0]))

            # check if some z_k vanished
            msg = ("Temporal component vanished, may be 'lbda' is too "
                   "high, please try to reduce its value.")
            check_if_vanished(z_hat, msg)

            # Update u
            B, C = _precompute_B_C(X, z_hat, H_hat, rois_idx)
            constants['C'] = C
            constants['B'] = B

            if get_time == 2:
                t0 = time.process_time()
            u_hat = _update_u(u_hat, constants)  # update
            if get_time == 2:
                ltime.append(time.process_time() - t0)

            if get_obj == 2:
                lobj.append(_obj(X=X, prox=prox_u_func, lbda=lbda, u=u_hat,
                                 z=z_hat, H=H_hat, rois_idx=rois_idx,
                                 valid=True))
                if verbose > 1:
                    if get_time:
                        print("[{}/{}] Update u done in {:.1f} s : "
                              "cost = {:.6f}".format(ii + 1, max_iter,
                                                     ltime[-1],
                                                     lobj[-1] / lobj[0]))
                    else:
                        print("[{}/{}] Update u done: "
                              "cost = {:.6f}".format(ii + 1, max_iter,
                                                     lobj[-1] / lobj[0]))

            if not deactivate_v_learning:

                # Update v
                constants['u'] = u_hat
                constants['z'] = z_hat

                if get_time == 2:
                    t0 = time.process_time()
                a_hat, v_hat = _update_v(a_hat, constants)  # update
                if get_time == 2:
                    ltime.append(time.process_time() - t0)

                if get_obj == 2:
                    lobj.append(_obj(X=X, prox=prox_u_func, lbda=lbda, u=u_hat,
                                     z=z_hat, H=H_hat, rois_idx=rois_idx,
                                     valid=True))
                    if verbose > 1:
                        if get_time:
                            print("[{}/{}] Update v done in {:.1f} s : "
                                  "cost = {:.6f}".format(ii + 1, max_iter,
                                                         ltime[-1],
                                                         lobj[-1] / lobj[0]))
                        else:
                            print("[{}/{}] Update v done: "
                                  "cost = {:.6f}".format(ii + 1, max_iter,
                                                         lobj[-1] / lobj[0]))

            if get_time == 1:
                ltime.append(time.process_time() - t0)

            if get_obj == 1:
                lobj.append(_obj(X=X, prox=prox_u_func, lbda=lbda, u=u_hat,
                                 z=z_hat, H=H_hat, rois_idx=rois_idx,
                                 valid=True))
                if verbose > 1:
                    if get_time:
                        print("[{}/{}] Iteration done in {:.1f} s : "
                              "cost = {:.6f}".format(ii + 1, max_iter,
                                                     ltime[-1],
                                                     lobj[-1] / lobj[0]))
                    else:
                        print("[{}/{}] Iteration done: "
                              "cost = {:.6f}".format(ii + 1, max_iter,
                                                     lobj[-1] / lobj[0]))

            if ii > 2 and get_obj:
                try:
                    check_obj(lobj, ii + 1, max_iter,
                              early_stopping=early_stopping,
                              raise_on_increase=raise_on_increase, eps=eps,
                              level=get_obj)
                except EarlyStopping as e:
                    if verbose > 1:
                        print(str(e))
                    break

    Dz_hat = np.diff(z_hat, axis=-1)

    if get_obj and get_time:
        return z_hat, Dz_hat, u_hat, a_hat, v_hat, v_init, lbda, \
               np.array(lobj), np.array(ltime)
    elif get_obj and not get_time:
        return z_hat, Dz_hat, u_hat, a_hat, v_hat, v_init, lbda, \
               np.array(lobj), None
    elif not get_obj and get_time:
        return z_hat, Dz_hat, u_hat, a_hat, v_hat, v_init, lbda, None, \
               np.array(ltime)
    else:
        return z_hat, Dz_hat, u_hat, a_hat, v_hat, v_init, lbda, None, None


def multi_runs_learn_u_z_v_multi(
        X, t_r, hrf_rois, hrf_model='scaled_hrf',
        deactivate_v_learning=False, n_atoms=10, n_times_atom=60,
        lbda_strategy='ratio', lbda=0.1, u_init_type='ica',
        prox_u='l1-positive-simplex', max_iter=100, get_obj=False,
        get_time=False, random_seed=None, early_stopping=True, eps=1.0e-5,
        raise_on_increase=True, n_jobs=1, nb_fit_try=1, verbose=0):
    """ Multiple initialization parallel running version of
    `learn_u_z_v_multi`.

    Parameters
    ----------
    X : array, shape (n_voxels, n_times), fMRI data
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    hrf_rois : dict (key: ROIs labels, value: indices of voxels of the ROI)
        atlas HRF
    hrf_model : str, (default='3_basis_hrf'), type of HRF model, possible
        choice are ['3_basis_hrf', '2_basis_hrf', 'scaled_hrf']
    deactivate_v_learning : bool, (default=False), option to force the
        estimated HRF to to the initial value.
    n_atoms : int, number of components on which to decompose the neural
        activity (number of temporal components and its associated spatial
        maps).
    n_times_atom : int, (default=30), number of points on which represent the
        Haemodynamic Response Function (HRF), this leads to the duration of the
        response function, duration = n_times_atom * t_r
    lbda_strategy str, (default='ratio'), strategy to fix the temporal
        regularization parameter, possible choice are ['ratio', 'fixed']
    lbda : float, (default=0.1), whether the temporal regularization parameter
        if lbda_strategy == 'fixed' or the ratio w.r.t lambda max if
        lbda_strategy == 'ratio'
    u_init_type : str, (default='ica'), strategy to init u, possible value are
        ['gaussian_noise', 'ica', 'patch']
    prox_u : str, (default='l2-positive-ball'), constraint to impose on the
        spatial maps possible choice are ['l2-positive-ball',
        'l1-positive-simplex']
    max_iter : int, (default=100), maximum number of iterations to perform the
        analysis
    get_obj : int, the level of cost-function saving, 0 to not compute it, 1 to
        compute it at each iteration, 2 to compute it at each estimation step
        (z, u, v)
    get_time : int, the level of computation-duration saving 0 to not compute
        it, 1 to compute it at each iteration, 2 to compute it at each
        estimation step
    random_seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the analysis
    early_stopping : bool, (default=True), whether to early stop the analysis
    eps : float, (default=1.0e-4), stoppping parameter w.r.t evolution of the
        cost-function
    raise_on_increase : bool, (default=True), whether to stop the analysis if
        the cost-function increases during an iteration. This can be due to the
        fact that the temporal regularization parameter is set to high
    nb_fit_try : int, (default=1), number of analysis to do with different
        initialization
    n_jobs : int, (default=1), the number of CPUs to use if multiple analysis
        with different initialization is done
    verbose : int, (default=0), verbose level, 0 no verbose, 1 low verbose,
        2 maximum verbose

    Return
    ------
    z : array, shape (n_atoms, n_times_valid), the estimated temporal
        components
    Dz : array, shape (n_atoms, n_times_valid), the estimated first order
        derivation temporal components
    u : array, shape (n_atoms, n_voxels), the estimated spatial maps
    a : array, shape (n_hrf_rois, n_param_HRF), the estimated HRF parameters
    v : array, shape (n_hrf_rois, n_times_atom), the estimated HRFs
    v : array, shape (n_hrf_rois, n_times_atom), the initial used HRFs
    lbda : float, the temporal regularization parameter used
    lobj : array or None, shape (n_iter,) or (3 * n_iter,), the saved
        cost-function
    ltime : array or None, shape (n_iter,) or(3 * n_iter,), the saved duration
        per steps

    Throws
    ------
    CostFunctionIncreased : if the cost-function increases during an iteration,
        of the analysis. This can be due to the fact that the temporal
        regularization parameter is set to high
    """
    params = dict(X=X, t_r=t_r, hrf_rois=hrf_rois, hrf_model=hrf_model,
                  deactivate_v_learning=deactivate_v_learning,
                  n_atoms=n_atoms, n_times_atom=n_times_atom,
                  lbda_strategy=lbda_strategy, lbda=lbda,
                  u_init_type=u_init_type, prox_u=prox_u,
                  max_iter=max_iter, get_obj=get_obj, get_time=get_time,
                  random_seed=random_seed, early_stopping=early_stopping,
                  eps=eps, raise_on_increase=raise_on_increase,
                  verbose=verbose)

    results = Parallel(n_jobs=n_jobs)(
                delayed(learn_u_z_v_multi)(**params)
                for _ in range(nb_fit_try))

    l_last_pobj = np.array([res[-2][-1] for res in results])
    best_run = np.argmin(l_last_pobj)

    if verbose > 0:
        print("[Decomposition] Best fitting: {}".format(best_run + 1))

    return results[best_run]
