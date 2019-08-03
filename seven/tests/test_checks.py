"""Testing module for the utility functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import pytest
import numpy as np
from seven.learn_u_z_v_multi import _update_z
from seven.checks import (check_random_state, check_len_hrf, check_if_vanished,
                          _get_lambda_max)
from seven.utils import _set_up_test


@pytest.mark.repeat(3)
def test_check_random_state():
    """ Test the check random state. """
    rng = check_random_state(None)
    assert isinstance(rng, np.random.RandomState)
    rng = check_random_state(np.random)
    assert isinstance(rng, np.random.RandomState)
    rng = check_random_state(3)
    assert isinstance(rng, np.random.RandomState)
    rng = check_random_state(check_random_state(None))
    assert isinstance(rng, np.random.RandomState)


@pytest.mark.repeat(3)
def test_check_len_hrf():
    """ Test the check HRF length. """
    length = 30
    assert len(check_len_hrf(np.empty(length - 1), length)) == length
    assert len(check_len_hrf(np.empty(length + 1), length)) == length
    assert len(check_len_hrf(np.empty(length), length)) == length


@pytest.mark.repeat(3)
def test_check_if_vanished():
    """ Test the check on vanished estimated vectors. """
    A = np.ones((10, 10))
    check_if_vanished(A)
    A = np.ones((10, 10))
    A[0, 0] = 0.0
    check_if_vanished(A)
    A = 1.0e-30 * np.ones((10, 10))
    pytest.raises(AssertionError, check_if_vanished, A=A)
    A = np.ones((10, 10))
    A[0, :] = 0.0
    pytest.raises(AssertionError, check_if_vanished, A=A)


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_get_lambda_max(seed):
    """ Test the lambda max estimation. """
    kwargs = _set_up_test(seed)
    z, u, H, v = kwargs['z'], kwargs['u'], kwargs['H'], kwargs['v']
    rois_idx, X = kwargs['rois_idx'], kwargs['X']
    lbda_max = _get_lambda_max(X, u, H, rois_idx)
    constants = dict(H=H, v=v, u=u, rois_idx=rois_idx, X=X, lbda=lbda_max)
    z_hat = _update_z(z, constants)
    assert np.linalg.norm(z_hat) / np.linalg.norm(z) < 0.05
