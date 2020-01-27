"""Testing module for the utility functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import pytest
from nilearn import datasets
from hemolearn import SLRDA


@pytest.mark.parametrize('verbose', [0, 1, 2])
@pytest.mark.parametrize('u_init_type', ['ica', 'patch', 'gaussian_noise'])
@pytest.mark.parametrize('hrf_model', ['3_basis_hrf', '2_basis_hrf',
                                       'scaled_hrf'])
def test_slrda(hrf_model, u_init_type, verbose):
    """ Test the main class: check that no Exception is raised """
    try:
        X = datasets.fetch_adhd(n_subjects=1).func[0]
        slrda = SLRDA(n_atoms=1, t_r=2.0, n_times_atom=30,
                      hrf_model=hrf_model, lbda=1.0e-3,  # no regu.
                      u_init_type=u_init_type, max_iter=1, eps=1.0-3,
                      raise_on_increase=True, random_state=None, n_jobs=1,
                      nb_fit_try=1, verbose=verbose)
        slrda.fit(X)
    except Exception as e:
        pytest.fail("Unexpected Exception raised: '{}'".format(e))
