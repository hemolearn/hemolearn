"""Testing module for the utility functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import pytest
from nilearn import datasets
from hemolearn import SLRDA


def test_slrda():
    """ Test the main class: check that no Exception is raised """
    try:
        X = datasets.fetch_adhd(n_subjects=1).func[0]
        slrda = SLRDA(n_atoms=5, t_r=2.0, n_times_atom=30,
                      hrf_atlas='scale007', hrf_model='3_basis_hrf',
                      lbda=5.0e-3, max_iter=2, eps=1.0-6,
                      raise_on_increase=True, random_state=0,
                      n_jobs=1, nb_fit_try=1, verbose=0)
        slrda.fit(X)
    except Exception as e:
        pytest.fail("Unexpected Exception raised: '{}'".format(e))
