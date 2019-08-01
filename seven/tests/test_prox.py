"""Testing module for the utility functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import pytest
import numpy as np
from seven.prox import _prox_positive_L2_ball
from seven.checks import check_random_state
from seven.tests.utils import _set_up


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_prox_positive_L2_ball(seed):
    """ Test the positive L2 ball proximal operator. """
    rng = check_random_state(seed)
    kwargs = _set_up(seed)
    u = kwargs['u']
    u_0 = u[0, :]
    prox_u_0 = _prox_positive_L2_ball(u_0)

    assert np.all(prox_u_0 >= 0.0)
    assert np.linalg.norm(prox_u_0) <= 1.0

    n_try = 100
    for _ in range(n_try):
        x = rng.randn(*u_0.shape)
        x[x < 0.0] = 0.0
        norm_x = x.ravel().dot(x.ravel())
        if not (norm_x <= 1.0):
            x /= norm_x
        assert np.linalg.norm(u_0 - prox_u_0) < np.linalg.norm(u_0 - x)
