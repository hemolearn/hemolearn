"""Testing module for the utility functions"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import pytest
from nilearn import datasets
from hemolearn import SLRDA


@pytest.mark.parametrize('hrf_model', ['3_basis_hrf', '2_basis_hrf',
                                       'scaled_hrf'])
@pytest.mark.parametrize('prox_z', ['tv', 'l1', 'l2', 'elastic-net'])
@pytest.mark.parametrize('prox_u', ['l2-positive-ball', 'l1-positive-simplex',
                                    'positive'])
def test_slrda_blank_run(hrf_model, prox_u, prox_z):
    """ Test the main class: check that no Exception is raised """
    try:
        X = datasets.fetch_adhd(n_subjects=1).func[0]
        slrda = SLRDA(n_atoms=1, t_r=2.0, n_times_atom=20,
                      hrf_model=hrf_model, lbda=1.0e-6,
                      lbda_strategy='fixed',  # no regu.
                      prox_u=prox_u, prox_z=prox_z,
                      max_iter=1, eps=1.0-3, raise_on_increase=True,
                      random_state=None, n_jobs=1, nb_fit_try=1)
        slrda.fit(X)
    except Exception as e:
        pytest.fail(f"Unexpected Exception raised: '{e}'")
