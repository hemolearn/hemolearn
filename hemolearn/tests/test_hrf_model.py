"""Testing module for the HRF models"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import pytest
import numpy as np
from scipy.stats import gamma
from hemolearn.checks import check_random_state
from hemolearn.hrf_model import (delta_derivative_double_gamma_hrf, _gamma_pdf,
                                 scaled_hrf, MIN_DELTA, MAX_DELTA)


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_delta_derivative_double_gamma_hrf(seed):
    t_r = 1.0
    rng = check_random_state(seed)
    eps = 1.0e-6
    delta = rng.uniform(MIN_DELTA, MAX_DELTA)
    grad = delta_derivative_double_gamma_hrf(delta, t_r)
    finite_grad = scaled_hrf(delta + eps, t_r) - scaled_hrf(delta, t_r)
    finite_grad /= eps
    np.testing.assert_allclose(finite_grad, grad, atol=1.0e-3)


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
@pytest.mark.parametrize('a', [6.0, 16.0])
@pytest.mark.parametrize('loc', [0.001])
@pytest.mark.parametrize('scale', [1.0])
def test_gamma_pdf(seed, a, loc, scale):
    """ Test the probability density function of the Gamma distribution """
    t_r = 1.0
    n_times_atom = 60
    t = np.linspace(0.0, t_r * n_times_atom, n_times_atom)
    ref_p = gamma.pdf(t, a=a, loc=loc, scale=scale)
    p = _gamma_pdf(t, a=a, loc=loc, scale=scale)
    np.testing.assert_allclose(ref_p, p)
