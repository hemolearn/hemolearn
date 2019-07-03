"""Seven package"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
from joblib import Parallel, delayed, Memory
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from nilearn import input_data

from .learn_u_z_multi import _update_u, _update_Lz, lean_u_z_multi
from .hrf_model import spm_hrf
from .atlas import fetch_atlas_basc_12_2015
from .utils import check_lbda


class SLRDM(TransformerMixin):
    """ BOLD Sparse Low-Rank Decomposition Model:
            - Distangle the neural activation and the haemodynamic
            contributions in the BOLD signal.
            - Compressed the neural activation signal in `n_atoms` main
            funtional networks
    """

    def __init__(self, n_atoms, t_r, v=None, n_times_atom=None,
                 lbda_strategy='ratio', lbda=0.1, hrf_atlas='basc',
                 max_iter=50, random_state=None, name="DL",
                 early_stopping=True, eps=1.0e-10, raise_on_increase=True,
                 smoothing_fwhm=6.0, detrend=True, standardize=True,
                 low_pass=0.1, high_pass=0.01, memory='.cache', n_jobs=1,
                 verbose=0):
        # model hyperparameters
        self.t_r = t_r
        self.n_atoms = n_atoms
        if v is None:
            msg = (" if v is not specified, n_times_atom should be given.")
            assert n_times_atom is not None, msg
            self.v = spm_hrf(self.t_r, n_times_atom)
        else:
            self.v = v
        self.n_times_atom = len(v)
        self.lbda_strategy = lbda_strategy
        self.lbda = lbda

        # HRF atlas
        if hrf_atlas == 'basc':
            self.brain_full_mask, self.hrf_atlas = fetch_atlas_basc_12_2015()
        else:
            raise ValueError("hrf_atlas should be in ['basc', ], "
                             "got {}".format(hrf_atlas))

        # fMRI data masker
        self.smoothing_fwhm = smoothing_fwhm
        self.detrend = detrend
        self.standardize = standardize
        self.low_pass = low_pass
        self.high_pass = high_pass

        # convergence parameters
        self.max_iter = max_iter
        self.random_state = random_state
        self.name = name
        self.early_stopping = early_stopping
        self.eps = eps
        self.raise_on_increase = raise_on_increase

        # technical parameters
        self.verbose = verbose
        self.n_jobs = n_jobs

        # model parameters
        self.Lz_hat_ = None
        self.z_hat_ = None
        self.u_hat_ = None
        self.lbda_ = lbda
        self.lobj_ = None
        self.ltime_ = None

    def fit(self, X, y=None, nb_fit_try=1, confounds=None):

        if not isinstance(X, str):
            raise ValueError("fMRI data to be decompose should passed as a "
                             "filename, instead of the raw data")

        self.masker_ = input_data.NiftiMasker(
                            mask_img=self.brain_full_mask, t_r=self.t_r,
                            smoothing_fwhm=self.smoothing_fwhm,
                            detrend=self.detrend, standardize=self.standardize,
                            low_pass=self.low_pass, high_pass=self.high_pass,
                            memory='.cache', memory_level=1, verbose=0)

        if confounds is not None:
            X = self.masker_.fit_transform(X, confounds=[confounds]).T
        else:
            X = self.masker_.fit_transform(X).T

        if self.verbose > 0:
            print("Data loaded shape: {} scans {} voxels".format(*X.shape))

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

        return self

    def transform(self, X):
        n_times_valid = X.shape[1] - self.n_times_atom + 1

        self._check_fitted()
        v_ = np.repeat(self.v[None, :], self.n_atoms, axis=0).squeeze()
        self.lbda_ = check_lbda(self.lbda_, self.lbda_strategy,
                                X, self.u_hat_, v_)
        constants = dict(lbda=self.lbda_, X=X, uv=np.c_[self.u_hat_, v_])
        Lz0 = np.zeros((self.n_atoms, n_times_valid))
        self.Lz_hat_ = _update_Lz(Lz0, constants)

        self.z_hat_ = np.diff(self.Lz_hat, axis=-1)
        self.lobj_ = None
        self.ltime_ = None

        return self

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
