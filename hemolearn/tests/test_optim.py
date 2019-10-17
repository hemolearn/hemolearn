"""Testing module for the utility functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import pytest
import numpy as np
from hemolearn.checks import check_random_state
from hemolearn.utils import lipschitz_est
from hemolearn.loss_grad import _obj
from hemolearn.optim import proximal_descent, cdclinmodel
from hemolearn.utils import _set_up_test
from hemolearn.prox import _prox_positive_l2_ball


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
@pytest.mark.parametrize('momentum', [None, 'fista', 'greedy'])
@pytest.mark.parametrize('restarting', [None, 'obj', 'descent'])
def test_proximal_descent(seed, momentum, restarting):
    """ Test the proximal descent algo. """
    rng = check_random_state(seed)
    m = 100
    x0 = rng.randn(m)
    y = rng.randn(int(m/2))
    A = rng.randn(int(m/2), m)
    AtA = A.T.dot(A)

    def obj(x):
        res = (A.dot(x) - y).ravel()
        return 0.5 * res.dot(res)

    def grad(x):
        return A.T.dot(A.dot(x) - y)

    def prox(x, step_size):
        return x

    def AtA(x):
        return A.T.dot(A).dot(x)
    step_size = 0.9 / lipschitz_est(AtA, x0.shape)

    params = dict(x0=x0, grad=grad, prox=prox, step_size=step_size,
                  momentum='fista', restarting='descent', max_iter=1000,
                  obj=obj, benchmark=True)
    x_hat, pobj, _ = proximal_descent(**params)

    assert pobj[0] > pobj[-1]

    if momentum is None:
        assert np.all(np.diff(pobj) < np.finfo(np.float64).eps)


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_cdclinmodel(seed):
    """ Test the coordinate descente on constraint linear model algo. """
    kwargs = _set_up_test(seed)
    u, C, B = kwargs['u'], kwargs['C'], kwargs['B']
    rois_idx = kwargs['rois_idx']
    X, z, v = kwargs['X'], kwargs['z'], kwargs['v']

    def _prox(u_k):
        return _prox_positive_l2_ball(u_k, 1.0)

    constants = dict(C=C, B=B, rois_idx=rois_idx, prox_u=_prox)

    def obj(u):
        return _obj(X=X, prox=_prox_positive_l2_ball, u=u, z=z,
                    rois_idx=rois_idx, v=v, valid=False, return_reg=False,
                    lbda=None)

    u_hat, pobj, _ = cdclinmodel(u, constants=constants, obj=obj,
                                 benchmark=True, max_iter=50)

    assert pobj[0] > pobj[-1]
