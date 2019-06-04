"""Utility module"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np

from alphacsc.utils.convolution import _dense_tr_conv_d


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
