"""Testing module for gradient and loss function"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import pytest
import numpy as np
from scipy.optimize import approx_fprime

from seven.convolution import make_toeplitz, adjconv_uv, adjconv_uH
from seven.checks import check_random_state
from seven.utils import _set_up_test


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_toeplitz(seed):
    """ Test the the making of the Toeplitz function. """
    rng = check_random_state(seed)
    n_times_atom, n_times_valid= 30, 100
    z = rng.randn(n_times_valid)
    v = rng.randn(n_times_atom)
    H = make_toeplitz(v, n_times_valid)
    np.testing.assert_allclose(np.convolve(v, z), H.dot(z))


@pytest.mark.repeat(3)
@pytest.mark.parametrize('seed', [None])
def test_adjconv_D(seed):
    """ Test the computation of uvtX and uHtX. """
    kwargs = _set_up_test(seed)
    u, v, rois_idx = kwargs['u'], kwargs['v'], kwargs['rois_idx']
    residual_i, H = kwargs['X'], kwargs['H'],
    uvtX = adjconv_uv(residual_i, u, v, rois_idx)
    uvtX_ = adjconv_uH(residual_i, u, H, rois_idx)

    # all HRFs / Toep. matrices are associated and all HRFs are equal, so we
    # only take the first: H[0, :, :]
    uvtX_ref = u.dot(residual_i).dot(H[0, :, :])

    np.testing.assert_allclose(uvtX_ref, uvtX)
    np.testing.assert_allclose(uvtX_ref, uvtX_)
