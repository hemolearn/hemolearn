"""Testing module for gradient and loss function"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np

from seven.hrf_model import spm_hrf
from seven.atlas import split_atlas
from seven.constants import _precompute_B_C
from seven.checks import check_random_state
from seven.convolution import make_toeplitz


def _set_up(seed):
    """ General set up function for the tests. """
    rng = check_random_state(None)
    t_r = 1.0
    n_hrf_rois = 2
    n_atoms, n_voxels = 4, 100
    n_times, n_times_atom = 100, 20
    n_times_valid = n_times - n_times_atom + 1
    labels = np.arange(n_hrf_rois, dtype=int)
    indices = np.split(np.arange(n_voxels, dtype=int), n_hrf_rois)
    hrf_rois = dict(zip(labels, indices))
    rois_idx, _, n_hrf_rois = split_atlas(hrf_rois)
    X = rng.randn(n_voxels, n_times)
    u = rng.randn(n_atoms, n_voxels)
    z = rng.randn(n_atoms, n_times_valid)
    v = np.r_[[spm_hrf(t_r=t_r, n_times_atom=n_times_atom)
               for _ in range(n_hrf_rois)]]
    H = np.empty((n_hrf_rois, n_times, n_times_valid))
    for m in range(n_hrf_rois):
        H[m, ...] = make_toeplitz(v[m], n_times_valid)
    B, C = _precompute_B_C(X, z, H, rois_idx)
    kwargs = dict(t_r=t_r, n_hrf_rois=n_hrf_rois, n_atoms=n_atoms,
                  n_voxels=n_voxels, n_times=n_times,
                  n_times_atom=n_times_atom, n_times_valid=n_times_valid,
                  rois_idx=rois_idx, X=X, z=z, u=u, H=H, v=v, B=B, C=C)
    return kwargs
