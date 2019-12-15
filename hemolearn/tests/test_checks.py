"""Testing module for the utility functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import pytest
import numpy as np
from hemolearn.learn_u_z_v_multi import _update_z
from hemolearn.checks import (check_random_state, check_len_hrf,
                              check_if_vanished, _get_lambda_max, check_obj,
                              EarlyStopping, CostFunctionIncreased, check_lbda)
from hemolearn.utils import _set_up_test


def test_check_lbda():
    """ Test the check on lbda. """
    with pytest.raises(ValueError):
        check_lbda(lbda=None, lbda_strategy='foo', X=None, u=None, H=None,
                   rois_idx=None)
    with pytest.raises(ValueError):
        check_lbda(lbda='foo', lbda_strategy='fixed', X=None, u=None, H=None,
                   rois_idx=None)


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
    with pytest.raises(ValueError):
        check_random_state('foo')


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
    constants = dict(H=H, v=v, u=u, rois_idx=rois_idx, X=X, lbda=lbda_max,
                     prox_z='tv', delta=2.0)
    z_hat = _update_z(z, constants)
    assert np.linalg.norm(z_hat) / np.linalg.norm(z) < 0.05


@pytest.mark.parametrize('level', [1, 2])
def test_check_obj(level):
    """ Test the cost-function check-function. """
    value_start = 0.0
    value_final = 100.0
    lobj = np.linspace(0.0, 100.0, int(value_final - value_start + 1))[::-1]

    # case 1: no exception
    check_obj(lobj=lobj, ii=2, max_iter=100, early_stopping=True,
              raise_on_increase=True, eps=np.finfo(np.float64).eps,
              level=level)

    # case 2: early stoppping exception
    with pytest.raises(EarlyStopping):
        check_obj(lobj=lobj, ii=2, max_iter=100, early_stopping=True,
                  raise_on_increase=True, eps=1.1, level=level)

    # case 3: cost-funcion raising exception
    with pytest.raises(CostFunctionIncreased):
        check_obj(lobj=lobj[::-1], ii=2, max_iter=100, early_stopping=True,
                  raise_on_increase=True, eps=np.finfo(np.float64).eps,
                  level=level)
