"""Seven package"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from nilearn import input_data

from .learn_u_z_multi import cached_multi_runs_learn_u_z_multi
from .atlas import fetch_atlas, split_atlas


class SLRDM(TransformerMixin):
    """ BOLD Sparse Low-Rank Semi Blind Deconvolution and Decomposition Model
        (BSLRSBDDM):
        - Distangle the neural activation and the haemodynamic
          contributions in the BOLD signal.
        - Compressed the neural activation signal in `n_atoms` main
          funtional networks
    """

    def __init__(self, n_atoms, t_r, n_times_atom=30, hrf_model='3_basis_hrf',
                 lbda_strategy='ratio', lbda=0.1, lbda_hrf=1.0,
                 hrf_atlas='basc-122', max_iter=50, random_state=None,
                 name="DL", early_stopping=True, eps=1.0e-4,
                 raise_on_increase=True, memory='.cache', n_jobs=1,
                 nb_fit_try=1, verbose=0):
        # model hyperparameters
        self.t_r = t_r
        self.n_atoms = n_atoms
        self.hrf_model = hrf_model
        self.n_times_atom = n_times_atom
        self.lbda_strategy = lbda_strategy
        self.lbda = lbda
        self.lbda_hrf = lbda_hrf

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
        self.memory = memory
        self.nb_fit_try = nb_fit_try

        # HRF atlas
        self.mask_full_brain, self.atlas_rois = fetch_atlas(hrf_atlas)
        self.hrf_rois = dict()

        # fMRI masker
        self.masker_ = input_data.NiftiMasker(mask_img=self.mask_full_brain,
                                              t_r=self.t_r, memory=self.memory,
                                              memory_level=1, verbose=0)

        # model parameters
        self.z_hat_ = None
        self.u_hat_ = None
        self.a_hat_ = None

        # derivated model parameters
        self.Dz_hat_ = None
        self.v_hat_ = None

        # auxilaries parameters
        self.roi_label_from_hrf_idx = None
        self.v_init_ = None
        self.lbda_ = lbda
        self.lobj_ = None
        self.ltime_ = None

    def fit(self, func_fname):

        if not isinstance(func_fname, str):
            raise ValueError("fMRI data to be decompose should passed as a "
                             "filename, instead of the raw data")

        X = self.masker_.fit_transform(func_fname).T

        rois = self.masker_.transform(self.atlas_rois).astype(int).ravel()
        index = np.arange(rois.shape[-1])
        for roi_label in np.unique(rois):
            self.hrf_rois[roi_label] = index[roi_label == rois]

        _, rois_label, _ = split_atlas(self.hrf_rois)
        self.roi_label_from_hrf_idx = rois_label

        if self.verbose > 0:
            print("Data loaded shape: {} voxels {} scans".format(*X.shape))

            if self.n_jobs > 1:
                print("Running {} fits in parallel on {} "
                      "CPU".format(self.nb_fit_try, self.n_jobs))
            else:
                print("Running {} fits in series".format(self.nb_fit_try))

        params = dict(X=X, t_r=self.t_r, hrf_rois=self.hrf_rois,
                      hrf_model=self.hrf_model, n_atoms=self.n_atoms,
                      n_times_atom=self.n_times_atom,
                      lbda_strategy=self.lbda_strategy, lbda=self.lbda,
                      lbda_hrf=self.lbda_hrf, max_iter=self.max_iter,
                      get_obj=1, get_time=1, u_init='random',
                      random_seed=self.random_state, name=self.name,
                      early_stopping=self.early_stopping, eps=self.eps,
                      raise_on_increase=self.raise_on_increase,
                      verbose=self.verbose, n_jobs=self.n_jobs,
                      nb_fit_try=self.nb_fit_try)

        res = cached_multi_runs_learn_u_z_multi(**params)

        self.z_hat_ = res[0]
        self.Dz_hat_ = res[1]
        self.u_hat_ = res[2]
        self.a_hat_ = res[3]
        self.v_hat_ = res[4]
        self.v_init_ = res[5]
        self.lbda_ = res[6]
        self.lobj_ = res[7]
        self.ltime_ = res[8]

        return self

    def fit_transform(self, X):
        self.fit(X)
        return self

    def transform(self, X, confounds=None):
        self._check_fitted()
        raise NotImplementedError("tranform method no implemented for now...")
        return self

    def _check_fitted(self):
        if self.u_hat is None:
            raise NotFittedError("Fit must be called before accessing the "
                                 "dictionary")

    @property
    def u_hat(self):
        return self.u_hat_

    @property
    def z_hat(self):
        self._check_fitted()
        return self.z_hat_

    @property
    def Dz_hat(self):
        self._check_fitted()
        return self.Dz_hat_

    @property
    def a_hat(self):
        self._check_fitted()
        return self.a_hat_

    @property
    def v_hat(self):
        self._check_fitted()
        return self.v_hat_

    @property
    def v_init(self):
        self._check_fitted()
        return self.v_init_

    @property
    def lobj(self):
        self._check_fitted()
        return self.lobj_

    @property
    def ltime(self):
        self._check_fitted()
        return self.ltime_
