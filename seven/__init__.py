"""Seven package"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
from joblib import Parallel, delayed, Memory
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError

from alphacsc.utils.hrf_model import spm_hrf

from .learn_u_z_multi import _update_u, _update_Lz, lean_u_z_multi


class SLRDM(TransformerMixin):
    """ BOLD Sparse Low-Rank Decomposition Model:
            - Distangle the neural activation and the haemodynamic
            contributions in the BOLD signal.
            - Compressed the neural activation signal in `n_atoms` main
            funtional networks
    """

    def __init__(self, n_atoms, v=None, t_r=None, n_times_atom=None,
                 lbda_strategy='ratio', lbda=0.1, max_iter=50,
                 random_state=None, name="DL", early_stopping=True,
                 eps=1.0e-10, raise_on_increase=True, memory='.cache',
                 n_jobs=1, verbose=0):
        # model hyperparameters
        self.n_atoms = n_atoms
        if v is None:
            msg = (" if v is not specified,"
                   "n_times_atom and t_r should be given.")
            assert n_times_atom is not None and t_r is not None, msg
            self.v = spm_hrf(t_r, n_times_atom)
        else:
            self.v = v
        self.lbda_strategy = lbda_strategy
        self.lbda = lbda

        # convergence parameters
        self.max_iter = max_iter
        self.random_state = random_state
        self.name = name
        self.early_stopping = early_stopping
        self.eps = eps
        self.raise_on_increase = raise_on_increase

        # technical parameters
        # self.memory = Memory(memory)
        self.verbose = verbose
        self.n_jobs = n_jobs

        # model parameters
        self.Lz_hat_ = None
        self.z_hat_ = None
        self.u_hat_ = None
        self.lbda_ = None
        self.lobj_ = None
        self.ltime_ = None

    def fit(self, X, y=None, nb_fit_try=1):
        if self.verbose > 0:
            if self.n_jobs > 1:
                print("Running {} fits in parallel on {} "
                     "CPU".format(nb_fit_try, self.n_jobs))
            else:
                print("Running {} fits in series".format(nb_fit_try))

        params = dict(X=X, v=self.v, n_atoms=self.n_atoms,
                      lbda_strategy=self.lbda_strategy, lbda=self.lbda,
                      max_iter=self.max_iter, get_obj=True, get_time=True,
                      random_state=self.random_state, name=self.name,
                      early_stopping=self.early_stopping, eps=self.eps,
                      raise_on_increase=self.raise_on_increase,
                      verbose=self.verbose)

        results = Parallel(n_jobs=self.n_jobs)(delayed(lean_u_z_multi)(**params)
                                               for _ in range(nb_fit_try))

        l_last_pobj = np.array([res[-2][-1] for res in results])
        best_run = np.argmin(l_last_pobj)
        res = results[best_run]

        self.Lz_hat_ = res[0]
        self.z_hat_ = res[1]
        self.u_hat_ = res[2]
        self.lbda_ = res[3]
        self.lobj_ = res[4]
        self.ltime_ = res[5]

        if self.verbose > 0:
            print("[{}] Best fitting: {}".format(self.name, best_run + 1))

        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        self.transform(X)

    def transform(self, X):
        self._check_fitted()

    def _check_fitted(self):
        if self.u_hat is None:
            raise NotFittedError("Fit must be called before accessing the "
                                 "dictionary")

    @property
    def u_hat(self):
        return self.u_hat_

    @property
    def Lz_hat(self):
        self._check_fitted()
        return self.Lz_hat_

    @property
    def z_hat(self):
        self._check_fitted()
        return self.z_hat_

    @property
    def v_hat(self):
        self._check_fitted()
        return self.v_hat_

    @property
    def lobj(self):
        self._check_fitted()
        return self.lobj_

    @property
    def ltime(self):
        self._check_fitted()
        return self.ltime_
