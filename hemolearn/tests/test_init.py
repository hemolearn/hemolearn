"""Testing module for the utility functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import pytest
from hemolearn.utils import _set_up_test
from hemolearn.deconvolution import init_z_hat, init_u_hat, init_v_hat


@pytest.mark.parametrize('z_init', [None, 'user_defined'])
def test_init_z(z_init):
    """ Test the z initialization. """
    seed = 0
    kwargs = _set_up_test(seed)
    n_subjects = 1
    rng = kwargs['rng']
    n_atoms = kwargs['n_atoms']
    n_times_valid = [kwargs['n_times_valid']]

    if z_init == 'user_defined':
        z_init = []
        for n in range(n_subjects):
            u_init_ = rng.randn(n_atoms, n_times_valid[n])
            z_init.append(u_init_)

    z_hat = init_z_hat(z_init, n_subjects, n_atoms, n_times_valid)

    for n in range(n_subjects):
        assert z_hat[n].shape == (n_atoms, n_times_valid[n])


@pytest.mark.parametrize('u_init_type', ['gaussian_noise', 'ica', 'patch',
                                         'user_defined'])
def test_u_init(u_init_type):
    """ Test the u initialization. """
    seed = 0
    kwargs = _set_up_test(seed)

    eta = 10.0
    n_spatial_maps = 1
    n_subjects = 1
    X = [kwargs['X']]
    v = [kwargs['v']]
    rng = kwargs['rng']
    n_atoms = kwargs['n_atoms']
    n_voxels = [kwargs['n_voxels']]
    n_times = [kwargs['n_times']]
    n_times_atom = kwargs['n_times_atom']

    if u_init_type == 'user_defined':
        u_init_type = []
        for n in range(n_subjects):
            u_init_ = rng.randn(n_atoms, n_voxels[n])
            u_init_type.append(u_init_)

    u_hat = init_u_hat(X, v, rng, u_init_type, eta, n_spatial_maps,
                       n_atoms, n_voxels, n_times, n_times_atom)

    for n in range(n_subjects):
        assert u_hat[n].shape == (n_atoms, n_voxels[n])


@pytest.mark.parametrize('hrf_model', ['3_basis_hrf', '2_basis_hrf',
                                       'scaled_hrf'])
def test_v_init(hrf_model):
    """ Test the v initialization. """
    seed = 0
    kwargs = _set_up_test(seed)

    constants = dict()
    delta_init = 1.0
    n_subjects = 1

    t_r = kwargs['t_r']
    n_times_atom = kwargs['n_times_atom']
    n_hrf_rois = kwargs['n_hrf_rois']

    v_hat, a_hat = init_v_hat(hrf_model, t_r, n_times_atom, n_subjects,
                              n_hrf_rois, constants, delta_init)

    assert len(v_hat) == n_subjects
    assert len(a_hat) == n_subjects

    for n in range(n_subjects):
        for m in range(n_hrf_rois):
            assert v_hat[n][m].shape == (n_times_atom,)
            if 'hrf_model' == '3_basis_hrf':
                assert a_hat[n][m].shape == (3,)
            elif 'hrf_model' == '2_basis_hrf':
                assert a_hat[n][m].shape == (2,)
            elif 'hrf_model' == 'scaled_hrf':
                assert a_hat[n][m].shape == (1,)
