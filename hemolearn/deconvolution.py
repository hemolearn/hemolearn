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
from .hrf_model import scaled_hrf, hrf_3_basis, hrf_2_basis
from .estim_v import _estim_v_d_basis, _estim_v_scaled_hrf
from .atlas import split_atlas
from .convolution import adjconv_uH, make_toeplitz
from .prox import (_prox_l1_simplex, _prox_positive_l2_ball, _prox_positive,
                   _prox_tv_multi)


def set_get_time_get_obj(verbose, raise_on_increase, early_stopping):
    """ Set 'get_time' and 'get_obj' from verbosity level.
    """
    if verbose >= 2:
        get_time = 2
        get_obj = 2

    elif verbose == 1:
        get_time = 1
        get_obj = 1

    else:
        get_time = 0
        get_obj = 0

    if (raise_on_increase or early_stopping) and (get_obj == 0):
        get_obj = 1
        print("If 'raise_on_increase' or 'early_stopping' are enables: "
            "'get_obj' is forced to '1'")

    return get_obj, get_time


def init_z_hat(z_init, n_subjects, n_atoms, n_times_valid):
    """ Initilization function of 'z_hat'
    """
    z_hat = []

    for n in range(n_subjects):
        if z_init is None:
            z_hat.append(np.zeros((n_atoms, n_times_valid[n])))

        else:
            if (n_atoms, n_times_valid[n]) != z_init[n].shape:
                raise ValueError(f"'z_init' should have the shape "
                                    f"(n_atoms, n_times_valid)="
                                    f"{(n_atoms, n_times_valid[n])}, "
                                    f"got {z_init[n].shape}")
            z_hat.append(np.copy(z_init[n]))

    return z_hat


def init_u_hat(X, v_hat, rng, u_init_type, eta, n_spatial_maps, n_atoms,
               n_voxels, n_times, n_times_atom):
    """ Initilization function of 'u_hat'
    """
    if u_init_type == 'gaussian_noise':
        u_hat = []
        for n in range(n_spatial_maps):
            u_hat_ = rng.randn(n_atoms, n_voxels[n])
            for k in range(n_atoms):
                u_hat_[k, :] = _prox_l1_simplex(u_hat_[k, :], eta=eta)
            u_hat.append(u_hat_)

    elif u_init_type == 'ica':
        ica = FastICA(n_components=n_atoms, algorithm='deflation', max_iter=50)
        ica.fit(X[0].T)
        u_hat_ = np.copy(ica.components_)
        del ica  # heavy object
        gc.collect()
        for k in range(n_atoms):
            u_hat_[k, :] = _prox_l1_simplex(u_hat_[k, :], eta=eta)
        u_hat = []
        for n in range(n_spatial_maps):
            u_hat.append(np.copy(u_hat_))

    elif u_init_type == 'patch':
        u_hat = []
        for n in range(n_spatial_maps):
            u_hat_ = np.empty((n_atoms, n_voxels[n]))
            for k in range(n_atoms):
                idx_start = rng.randint(0, n_times[n] - n_times_atom)
                idx_stop = idx_start + n_times_atom
                u_k_init = X[n][:, idx_start:idx_stop].dot(v_hat[n][0, :])
                u_hat_[k, :] = _prox_l1_simplex(u_k_init, eta=eta)
            u_hat.append(u_hat_)

    elif (isinstance(u_init_type, list)
          and all([u_init_type[i].shape == (n_atoms, n_voxels_)
                   for i, n_voxels_ in enumerate(n_voxels)])):
        u_hat = u_init_type

    else:
        raise ValueError(f"u_init_type should be in ['ica', "
                        f"'gaussian_noise', 'patch'],"
                        f" got {u_init_type}")

    return u_hat


def init_v_hat(hrf_model, t_r, n_times_atom, n_subjects, n_hrf_rois,
                constants, delta_init):
    """ Initilization function of 'v_hat'
    """
    if hrf_model == '3_basis_hrf':
        h = hrf_3_basis(t_r, n_times_atom)
        v_hat, a_hat = [], []
        for _ in range(n_subjects):
            a_hat_ = np.c_[[np.array([1.0, 0.0, 0.0])
                            for _ in range(n_hrf_rois)]]
            a_hat.append(a_hat_)
            v_hat.append(np.c_[[a_.dot(h) for a_ in a_hat_]])
        constants['h'] = h

    elif hrf_model == '2_basis_hrf':
        h = hrf_2_basis(t_r, n_times_atom)
        v_hat, a_hat = [], []
        for _ in range(n_subjects):
            a_hat_ = np.c_[[np.array([1.0, 0.0]) for _ in range(n_hrf_rois)]]
            a_hat.append(a_hat_)
            v_hat.append(np.c_[[a_.dot(h) for a_ in a_hat_]])
        constants['h'] = h

    elif hrf_model == 'scaled_hrf':
        v_ref = scaled_hrf(delta=delta_init, t_r=t_r,
                        n_times_atom=n_times_atom)
        v_hat, a_hat = [], []
        for _ in range(n_subjects):
            a_hat.append(delta_init * np.ones(n_hrf_rois))
            v_hat.append(np.c_[[v_ref for _ in range(n_hrf_rois)]])
        constants['t_r'] = t_r
        constants['n_times_atom'] = n_times_atom

    else:
        raise ValueError(f"hrf_model should be in ['3_basis_hrf', "
                        f"'2_basis_hrf', 'scaled_hrf', 'fir_hrf'], "
                        f"got {hrf_model}")

    return v_hat, a_hat


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
        raise ValueError(f"hrf_model should be in ['3_basis_hrf', "
                         f"'2_basis_hrf', 'scaled_hrf', 'fir_hrf'], "
                         f"got {hrf_model}")


def _update_u(u0, constants):  # pragma: no cover
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


def _update_z(z0, constants):  # pragma: no cover
    """ Update the temporal components.

    Parameters
    ----------
    z0 : array, shape (n_atoms, n_voxels), initial temporal components
    constants : dict, gather the usefull constant for the estimation of the
        temporal components, keys are (v, H, u, rois_idx, X, lbda, prox_z)

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
    msg = "'rho' is missing in 'constants' for '_update_z' step."
    assert ('rho' in constants), msg
    msg = "'prox_z' is missing in 'constants' for '_update_z' step."
    assert ('prox_z' in constants), msg

    u = constants['u']
    H = constants['H']
    v = constants['v']
    rois_idx = constants['rois_idx']
    X = constants['X']
    lbda = constants['lbda']
    rho = constants['rho']
    prox_z = constants['prox_z']

    uvtuv = _precompute_uvtuv(u=u, v=v, rois_idx=rois_idx)
    uvtX = adjconv_uH(X, u=u, H=H, rois_idx=rois_idx)

    if prox_z == 'tv':
        def prox(z, step_size):
            return _prox_tv_multi(z, lbda, step_size)

    elif prox_z == 'l1':
        def prox(z, step_size):
            th = lbda * step_size
            return np.sign(z) * np.clip(np.abs(z) - th, 0.0, None)

    elif prox_z == 'l2':
        def prox(z, step_size):
            th = lbda * step_size
            return np.clip(1 - th / np.linalg.norm(z), 0.0, None) * z

    elif prox_z == 'elastic-net':
        def prox(z, step_size):
            th = lbda * step_size
            prox_z = np.sign(z) * np.clip(np.abs(z) - th, 0.0, None)
            return prox_z / (1.0 + th * rho)

    else:
        raise ValueError(f"'prox_z' should be in ['tv', 'l1', 'l2', "
                         f"'elastic-net'], got {prox_z}")

    def grad(z):
        return _grad_z(z, uvtuv, uvtX)

    def AtA(z):
        return _grad_z(z, uvtuv)
    step_size = 0.9 / lipschitz_est(AtA, z0.shape)

    params = dict(x0=z0, grad=grad, prox=prox, step_size=step_size,
                  momentum='fista', restarting='descent', max_iter=100)
    z_hat = proximal_descent(**params)

    return z_hat


def blind_deconvolution_multiple_subjects(
        X, t_r, hrf_rois, hrf_model='scaled_hrf', shared_spatial_maps=False,
        deactivate_v_learning=False, deactivate_z_learning=False,
        deactivate_u_learning=False, n_atoms=10, n_times_atom=60, prox_z='tv',
        lbda_strategy='ratio', lbda=0.1, rho=2.0, delta_init=1.0,
        u_init_type='ica', eta=10.0, z_init=None, prox_u='l1-positive-simplex',
        max_iter=100, get_obj=0, get_time=0, random_seed=None,
        early_stopping=True, eps=1.0e-5, raise_on_increase=True, verbose=0):
    """ Multivariate Blind Deconvolution main function for mulitple subjects.

    Parameters
    ----------
    X : array, shape (n_voxels, n_times), fMRI data
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    hrf_rois : dict (key: ROIs labels, value: indices of voxels of the ROI)
        atlas HRF
    hrf_model : str, (default='3_basis_hrf'), type of HRF model, possible
        choice are ['3_basis_hrf', '2_basis_hrf', 'scaled_hrf']
    shared_spatial_maps : bool, whether or not to learn a single set of
        spatials maps accross subjects.
    deactivate_v_learning : bool, (default=False), option to force the
        estimated HRF to to the initial value.
    deactivate_z_learning : bool, (default=False), option to force the
        estimated z to its initial value.
    deactivate_u_learning : bool, (default=False), option to force the
        estimated u to its initial value.
    n_atoms : int, number of components on which to decompose the neural
        activity (number of temporal components and its associated spatial
        maps).
    n_times_atom : int, (default=30), number of points on which represent the
        Haemodynamic Response Function (HRF), this leads to the duration of the
        response function, duration = n_times_atom * t_r
    prox_z : str, (default='tv'), temporal proximal operator should be in
        ['tv', 'l1', 'l2', 'elastic-net']
    lbda_strategy str, (default='ratio'), strategy to fix the temporal
        regularization parameter, possible choice are ['ratio', 'fixed']
    lbda : float, (default=0.1), whether the temporal regularization parameter
        if lbda_strategy == 'fixed' or the ratio w.r.t lambda max if
        lbda_strategy == 'ratio'
    rho : float, (default=2.0), the elastic-net temporal regularization
        parameter
    delta_init : float, (default=1.0), the initialization value for the HRF
        dilation parameter
    u_init_type : str, (default='ica'), strategy to init u, possible value are
        ['gaussian_noise', 'ica', 'patch']
    eta : float, (default=10.0), the spatial sparsity regularization parameter
    z_init : None or array, (default=None), initialization of z, if None, z is
        initialized to zero
    prox_u : str, (default='l2-positive-ball'), constraint to impose on the
        spatial maps possible choice are ['l2-positive-ball',
        'l1-positive-simplex', 'positive']
    max_iter : int, (default=100), maximum number of iterations to perform the
        analysis
    random_seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the analysis
    early_stopping : bool, (default=True), whether to early stop the analysis
    eps : float, (default=1.0e-4), stoppping parameter w.r.t evolution of the
        cost-function
    raise_on_increase : bool, (default=True), whether to stop the analysis if
        the cost-function increases during an iteration. This can be due to the
        fact that the temporal regularization parameter is set to high
    verbose : int, (default=0), verbose level, 0 no verbose, 1 low verbose,
        2 (or more) maximum verbose

    Return
    ------
    z : array, shape (n_subjects, n_atoms, n_times_valid), the estimated
        temporal components
    Dz : array, shape (n_subjects, n_atoms, n_times_valid), the estimated first
        order derivation temporal components
    u : array, shape (n_subjects or 1, n_atoms, n_voxels), the estimated
        spatial maps
    a : array, shape (n_subjects, n_hrf_rois, n_param_HRF), the estimated HRF
        parameters
    v : array, shape (n_subjects, n_hrf_rois, n_times_atom), the estimated HRFs
    v : array, shape (n_subjects, n_hrf_rois, n_times_atom), the initial used
        HRFs
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
    if isinstance(X, np.ndarray) and X.ndim == 2:
        X = [X]  # handle single subject case

    if not isinstance(X, list):  # break not valid cases
        raise ValueError("Subjects data X should a list.")

    n_subjects = len(X)

    for n in range(n_subjects):
        X[n] = X[n].astype(np.float64)

    if verbose > 1:
        print(f"[SLRDA] Seed used = {random_seed}")

    rng = check_random_state(random_seed)

    n_times, n_times_valid, n_voxels = [], [], []
    for n in range(n_subjects):

        n_voxels_, n_times_ = X[n].shape
        n_times_valid_ = n_times_ - n_times_atom + 1

        if n_times_valid_ < 2 * n_times_atom - 1:
            raise ValueError("'n_times_atom' is too hight w.r.t the duration "
                             "of the acquisition, please reduce it.")

        n_voxels.append(n_voxels_)
        n_times.append(n_times_)
        n_times_valid.append(n_times_valid_)

    for n in range(1, n_subjects):
        if n_voxels[n] != n_voxels[0]:
            raise ValueError("All subjects do not have the same number of "
                             "voxels.")

    if n_subjects == 1 and shared_spatial_maps:
        print("Only 1 subject loaded, 'shared_spatial_maps' force to False")

    if (deactivate_v_learning
        and deactivate_u_learning
        and deactivate_z_learning):
        raise ValueError("'deactivate_v_learning', 'deactivate_z_learning' "
                         "and 'deactivate_u_learning' can't be set to True "
                         "all together.")

    if deactivate_z_learning:
        prox_u = 'positive'
        print("'deactivate_z_learning' is enable: 'prox_u' is forced to "
              "'positive'")

    if deactivate_z_learning and (z_init is None):
        raise ValueError("If 'deactivate_z_learning' is enable 'z_init' should"
                         " be provided")

    # split atlas
    rois_idx, _, n_hrf_rois = split_atlas(hrf_rois)

    constants = dict(rois_idx=rois_idx, hrf_model=hrf_model)

    get_obj, get_time = set_get_time_get_obj(verbose, raise_on_increase,
                                             early_stopping)

    # v initialization
    v_hat, a_hat = init_v_hat(hrf_model, t_r, n_times_atom, n_subjects,
                              n_hrf_rois, constants, delta_init)
    v_init = v_hat[0][0, :]

    # H initialization
    H_hat = []
    for n in range(n_subjects):
        H_hat_ = np.empty((n_hrf_rois, n_times[n], n_times_valid[n]))
        for m in range(n_hrf_rois):
            H_hat_[m, :, :] = make_toeplitz(v_hat[n][m],
                                            n_times_valid=n_times_valid[n])
        H_hat.append(H_hat_)

    # z initialization
    z_hat = init_z_hat(z_init, n_subjects, n_atoms, n_times_valid)

    n_spatial_maps = 1 if shared_spatial_maps else n_subjects

    # u initialization
    u_hat = init_u_hat(X, v_hat, rng, u_init_type, eta, n_spatial_maps, n_atoms,
                       n_voxels, n_times, n_times_atom)

    # temporal regularization parameter
    lbda_new = []
    for n in range(n_subjects):
        u_idx = 0 if shared_spatial_maps else n
        lbda_ = check_lbda(lbda, lbda_strategy, X[n], u_hat[u_idx], H_hat[n],
                           rois_idx, prox_z)
        lbda_new.append(lbda_)
    lbda = lbda_new

    # spatial regularization parameter
    if prox_u == 'l2-positive-ball':
        def _prox(u_k):
            return _prox_positive_l2_ball(u_k, step_size=1.0)
        prox_u_func = _prox
    elif prox_u == 'l1-positive-simplex':
        def _prox(u_k):
            return _prox_l1_simplex(u_k, eta=eta)
        prox_u_func = _prox
    elif prox_u == 'positive':
        def _prox(u_k):
            return _prox_positive(u_k, step_size=1.0)
        prox_u_func = _prox
    else:
        raise ValueError(f"prox_u should be in ['l2-positive-ball', "
                         f"'l1-positive-simplex', 'positive'], got {prox_u}")

    constants['prox_z'] = prox_z
    constants['rho'] = rho
    constants['prox_u'] = prox_u_func

    if get_obj:
        _obj_value = 0.0
        for n in range(n_subjects):
            u_idx = 0 if shared_spatial_maps else n
            _obj_value += _obj(X=X[n], prox=prox_u_func, lbda=lbda[n],
                               u=u_hat[u_idx], z=z_hat[n], H=H_hat[n],
                               rois_idx=rois_idx, valid=True, rho=rho,
                               prox_z=prox_z) / n_subjects
        lobj = [_obj_value]

    if get_time:
        ltime = [0.0]

    # main loop
    with threadpool_limits(limits=1):

        for ii in range(max_iter):

            if get_time == 1:
                t0 = time.process_time()

            # use Toeplitz matrices for obj. func. computation (Numpy speed-up)
            for n in range(n_subjects):
                for m in range(n_hrf_rois):
                    H_hat[n][m, :, :] = make_toeplitz(
                                v_hat[n][m], n_times_valid=n_times_valid[n])

            if not deactivate_z_learning:

                if get_time == 2:
                    t0 = time.process_time()

                # Update z
                z_hat_new = []
                for n in range(n_subjects):
                    u_idx = 0 if shared_spatial_maps else n
                    constants['a'] = a_hat[n]
                    constants['v'] = v_hat[n]
                    constants['H'] = H_hat[n]
                    constants['u'] = u_hat[u_idx]
                    constants['X'] = X[n]
                    constants['lbda'] = lbda[n]
                    z_hat_new.append(_update_z(z_hat[n], constants))  # update
                z_hat = z_hat_new

                if get_time == 2:
                    ltime.append(time.process_time() - t0)

                if get_obj == 2:
                    _obj_value = 0.0
                    for n in range(n_subjects):
                        u_idx = 0 if shared_spatial_maps else n
                        _obj_value_ = _obj(X=X[n], prox=prox_u_func,
                                           lbda=lbda[n], u=u_hat[u_idx],
                                           z=z_hat[n], H=H_hat[n],
                                           rois_idx=rois_idx, valid=True,
                                           rho=rho, prox_z=prox_z)
                        _obj_value += _obj_value_ / n_subjects
                    lobj.append(_obj_value)

                    if verbose > 1:
                        if get_time:
                            print(f"[{ii + 1:03d}/{max_iter:03d}][001/003] "
                                  f"Temporal activations estimation done in "
                                  f"{ltime[-1]:.3f}s: cost = "
                                  f"{lobj[-1] / lobj[0]:.6f}% (of "
                                  f"initial value)")
                        else:
                            print(f"[{ii + 1:03d}/{max_iter:03d}][1/3] "
                                  f"Temporal activations estimation done: "
                                  f"cost = {lobj[-1] / lobj[0]:.6f}% "
                                  f"(of initial value)")

            # check if some z_k vanished
            msg = ("Temporal component vanished, may be 'lbda' is too "
                   "high, please try to reduce its value.")
            for n in range(n_subjects):
                check_if_vanished(z_hat[n], msg)

            if not deactivate_u_learning:

                if get_time == 2:
                    t0 = time.process_time()

                # Update u
                u_hat_new = []
                if shared_spatial_maps:
                    B, C = [], []
                    for n in range(n_subjects):
                        B_, C_ = _precompute_B_C(X[n], z_hat[n], H_hat[n],
                                                 rois_idx)
                        B.append(B_)
                        C.append(C_)
                    constants['C'] = np.mean(C, axis=0)
                    constants['B'] = np.mean(B, axis=0)
                    u_hat[0] = _update_u(u_hat[0], constants)  # update
                else:
                    for n in range(n_subjects):
                        B, C = _precompute_B_C(X[n], z_hat[n], H_hat[n],
                                               rois_idx)
                        constants['C'] = C
                        constants['B'] = B
                        # update
                        u_hat_new.append(_update_u(u_hat[n], constants))
                    u_hat = u_hat_new

                if get_time == 2:
                    ltime.append(time.process_time() - t0)

                if get_obj == 2:
                    _obj_value = 0.0
                    for n in range(n_subjects):
                        u_idx = 0 if shared_spatial_maps else n
                        _obj_value_ = _obj(X=X[n], prox=prox_u_func,
                                           lbda=lbda[n], u=u_hat[u_idx],
                                           z=z_hat[n], H=H_hat[n],
                                           rois_idx=rois_idx, valid=True,
                                           rho=rho, prox_z=prox_z)
                        _obj_value += _obj_value_ / n_subjects
                    lobj.append(_obj_value)

                    if verbose > 1:
                        if get_time:
                            print(f"[{ii + 1:03d}/{max_iter:03d}][002/003] "
                                  f"Spatial maps estimation         done in "
                                  f"{ltime[-1]:.3f}s: cost = "
                                  f"{lobj[-1] / lobj[0]:.6f}% (of "
                                  f"initial value)")
                        else:
                            print(f"[{ii + 1:03d}/{max_iter:03d}][002/003] "
                                  f"Spatial maps estimation  done: cost = "
                                  f"{lobj[-1] / lobj[0]:.6f}% "
                                  f"(of initial value)")

            if not deactivate_v_learning:

                if get_time == 2:
                    t0 = time.process_time()

                # Update v
                a_hat_new, v_hat_new = [], []
                for n in range(n_subjects):
                    u_idx = 0 if shared_spatial_maps else n
                    constants['u'] = u_hat[u_idx]
                    constants['z'] = z_hat[n]
                    constants['X'] = X[n]
                    a_hat_, v_hat_ = _update_v(a_hat[n], constants)  # update
                    a_hat_new.append(a_hat_)
                    v_hat_new.append(v_hat_)
                a_hat, v_hat = a_hat_new, v_hat_new

                if get_time == 2:
                    ltime.append(time.process_time() - t0)

                if get_obj == 2:
                    _obj_value = 0.0
                    for n in range(n_subjects):
                        u_idx = 0 if shared_spatial_maps else n
                        _obj_value_ = _obj(X=X[n], prox=prox_u_func,
                                           lbda=lbda[n], u=u_hat[u_idx],
                                           z=z_hat[n], H=H_hat[n],
                                           rois_idx=rois_idx, valid=True,
                                           rho=rho, prox_z=prox_z)
                        _obj_value += _obj_value_ / n_subjects
                    lobj.append(_obj_value)

                    if verbose > 1:
                        if get_time:
                            print(f"[{ii + 1:03d}/{max_iter:03d}][003/003] "
                                  f"HRF estimation                  done in "
                                  f"{ltime[-1]:.3f}s: cost = "
                                  f"{lobj[-1] / lobj[0]:.6f}% (of "
                                  f"initial value)")
                        else:
                            print(f"[{ii + 1:03d}/{max_iter:03d}][003/003] "
                                  f"HRF estimation done:           cost = "
                                  f"{lobj[-1] / lobj[0]:.6f}% "
                                  f"(of initial value)")

            if get_time == 1:
                ltime.append(time.process_time() - t0)

            if get_obj == 1:
                _obj_value = 0.0
                for n in range(n_subjects):
                    u_idx = 0 if shared_spatial_maps else n
                    _obj_value += _obj(X=X[n], prox=prox_u_func, lbda=lbda[n],
                                       u=u_hat[u_idx], z=z_hat[n], H=H_hat[n],
                                       rois_idx=rois_idx, valid=True,
                                       rho=rho, prox_z=prox_z) / n_subjects
                lobj.append(_obj_value)

                if verbose == 1:
                    if get_time:
                        print(f"[{ii + 1:03d}/{max_iter:03d}] Iteration done "
                              f"in {ltime[-1]:.3f}s: cost = "
                              f"{lobj[-1] / lobj[0]:.6f}% (of initial value)")
                    else:
                        print(f"[{ii + 1:03d}/{max_iter:03d}] Iteration done: "
                              f"cost = {lobj[-1] / lobj[0]:.6f}% (of "
                              f"initial value)")

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

    Dz_hat = [np.diff(z_hat[n], axis=-1) for n in range(n_subjects)]

    return_vars = [z_hat, Dz_hat, u_hat, a_hat, v_hat, v_init, lbda]

    if get_obj and get_time:
        return_vars.extend([lobj, ltime])
        return return_vars

    elif get_obj and not get_time:
        return_vars.extend([lobj, None])
        return return_vars

    elif not get_obj and get_time:
        return_vars.extend([None, ltime])
        return return_vars

    else:
        return_vars.extend([None, None])
        return return_vars


def multi_runs_blind_deconvolution_multiple_subjects(
        X, t_r, hrf_rois, hrf_model='scaled_hrf', shared_spatial_maps=False,
        deactivate_v_learning=False, deactivate_z_learning=False,
        deactivate_u_learning=False, n_atoms=10, n_times_atom=60, prox_z='tv',
        lbda_strategy='ratio', lbda=0.1, rho=2.0, delta_init=1.0,
        u_init_type='ica', eta=10.0, z_init=None, prox_u='l1-positive-simplex',
        max_iter=100, get_obj=0, get_time=0, random_seed=None,
        early_stopping=True, eps=1.0e-5, raise_on_increase=True, n_jobs=1,
        nb_fit_try=1, verbose=0):
    """ Multiple initialization parallel running version of
    `blind_deconvolution_single_subject`.

    Parameters
    ----------
    X : array, shape (n_voxels, n_times), fMRI data
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    hrf_rois : dict (key: ROIs labels, value: indices of voxels of the ROI)
        atlas HRF
    hrf_model : str, (default='3_basis_hrf'), type of HRF model, possible
        choice are ['3_basis_hrf', '2_basis_hrf', 'scaled_hrf']
    shared_spatial_maps : bool, whether or not to learn a single set of
        spatials maps accross subjects.
    deactivate_v_learning : bool, (default=False), option to force the
        estimated HRF to to the initial value.
    deactivate_z_learning : bool, (default=False), option to force the
        estimated z to its initial value.
    deactivate_u_learning : bool, (default=False), option to force the
        estimated u to its initial value.
    n_atoms : int, number of components on which to decompose the neural
        activity (number of temporal components and its associated spatial
        maps).
    n_times_atom : int, (default=30), number of points on which represent the
        Haemodynamic Response Function (HRF), this leads to the duration of the
        response function, duration = n_times_atom * t_r
    prox_z : str, (default='tv'), temporal proximal operator should be in
        ['tv', 'l1', 'l2', 'elastic-net']
    lbda_strategy str, (default='ratio'), strategy to fix the temporal
        regularization parameter, possible choice are ['ratio', 'fixed']
    lbda : float, (default=0.1), whether the temporal regularization parameter
        if lbda_strategy == 'fixed' or the ratio w.r.t lambda max if
        lbda_strategy == 'ratio'
    rho : float, (default=2.0), the elastic-net temporal regularization
        parameter
    delta_init : float, (default=1.0), the initialization value for the HRF
        dilation parameter
    u_init_type : str, (default='ica'), strategy to init u, possible value are
        ['gaussian_noise', 'ica', 'patch']
    eta : float, (default=10.0), the spatial sparsity regularization parameter
    z_init : None or array, (default=None), initialization of z, if None, z is
        initialized to zero
    prox_u : str, (default='l2-positive-ball'), constraint to impose on the
        spatial maps possible choice are ['l2-positive-ball',
        'l1-positive-simplex', 'positive']
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
                  shared_spatial_maps=shared_spatial_maps,
                  deactivate_v_learning=deactivate_v_learning,
                  deactivate_z_learning=deactivate_z_learning,
                  deactivate_u_learning=deactivate_u_learning,
                  n_atoms=n_atoms, n_times_atom=n_times_atom,
                  prox_z=prox_z, lbda_strategy=lbda_strategy, lbda=lbda,
                  rho=rho, delta_init=delta_init, u_init_type=u_init_type,
                  eta=eta, z_init=z_init, prox_u=prox_u, max_iter=max_iter,
                  get_obj=get_obj, get_time=get_time, random_seed=random_seed,
                  early_stopping=early_stopping, eps=eps,
                  raise_on_increase=raise_on_increase, verbose=verbose)

    results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(blind_deconvolution_multiple_subjects)(**params)
                for _ in range(nb_fit_try))

    l_last_pobj = np.array([res[-2][-1] for res in results])
    best_run = np.argmin(l_last_pobj)

    if verbose > 0:
        print(f"[Decomposition] Best fitting: {best_run + 1}")

    return results[best_run]
