""" Seven is a Python module to estimate the Haemodynamic Response Function
(HRF) in brain from resting-state or task fMRI data (BOLD signal). It relies on
a Sparse Low-Rank Deconvolution Analysis (SLRDA) to distangles the
neurovascular coupling from the the neural activity.

The advantages of SLRDA are:

* The estimation of the HRF for each voxels of the brain.
* The decomposition of the neural activity in a set of temporal components and
its associated spatial map that describe a function network in the brain.

The disadvantages of SLRDA include:

* If the temporal resolution in the fMRI data is too low (TR > 2s) it's likely
that the analysis will not found major difference between the HRFs. This is due
to the fact that the common time-to-peak difference between HRFs within the
same brain varies at maximum of 2 s, if the temporal resolution is greater that
this, no effect will be estimated.
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
from joblib import Memory
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from nilearn import input_data

from .learn_u_z_v_multi import multi_runs_learn_u_z_v_multi
from .atlas import fetch_atlas_basc_2015, split_atlas
from .build_numba import build_numba_functions_of_seven


build_numba_functions_of_seven()


class SLRDA(TransformerMixin):
    """ Sparse Low-Rank Deconvolution Analysis (SLRDA) is a method to distangle
    the neural activation and the haemodynamic contributions in the fMRI data
    (BOLD signal).

    Parameters
    ----------
    n_atoms : int, number of components on which to decompose the neural
        activity (number of temporal components and its associated spatial
        maps).
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    n_times_atom : int, (default=30), number of points on which represent the
        Haemodynamic Response Function (HRF), this leads to the duration of the
        response function, duration = n_times_atom * t_r
    hrf_model : str, (default='3_basis_hrf'), type of HRF model, possible
        choice are ['3_basis_hrf', '2_basis_hrf', 'scaled_hrf']
    deactivate_v_learning : bool, (default=False), option to force the
        estimated HRF to to the initial value.
    lbda_strategy str, (default='ratio'), strategy to fix the temporal
        regularization parameter, possible choice are ['ratio', 'fixed']
    lbda : float, (default=0.1), whether the temporal regularization parameter
        if lbda_strategy == 'fixed' or the ratio w.r.t lambda max if
        lbda_strategy == 'ratio'
    u_init_type : str, (default='ica'), strategy to init u, possible value are
        ['gaussian_noise', 'ica', 'patch']
    prox_u : str, (default='l2-positive-ball'), constraint to impose on the
        spatial maps possible choice are ['l2-positive-ball',
        'l1-positive-simplex']
    hrf_atlas : str, (default='basc122'), HRF atlas to use to define the HRF
        ROIs, possible choice are ['scale007', 'scale012', 'scale036',
        'scale064', 'scale122']
    max_iter : int, (default=100), maximum number of iterations to perform the
        analysis
    random_state : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the analysis
    early_stopping : bool, (default=True), whether to early stop the analysis
    eps : float, (default=1.0e-4), stoppping parameter w.r.t evolution of the
        cost-function
    raise_on_increase : bool, (default=True), whether to stop the analysis if
        the cost-function increases during an iteration. This can be due to the
        fact that the temporal regularization parameter is set to high
    cache_dir : str or None, (default='.cache'), the directory name to
        cache the analysis, if None the decomposition is not cached, if a
        string is given or let to default, caching is done
    nb_fit_try : int, (default=1), number of analysis to do with different
        initialization
    n_jobs : int, (default=1), the number of CPUs to use if multiple analysis
        with different initialization is done
    verbose : int, (default=0), verbose level, 0 no verbose, 1 low verbose,
        2 maximum verbose

    Throws
    ------
    CostFunctionIncreased : if the cost-function increases during an iteration,
        of the analysis. This can be due to the fact that the temporal
        regularization parameter is set to high
    """

    def __init__(self, n_atoms, t_r, n_times_atom=60, hrf_model='scaled_hrf',
                 lbda_strategy='ratio', lbda=0.1, u_init_type='ica',
                 prox_u='l1-positive-simplex', hrf_atlas='scale122',
                 deactivate_v_learning=False, max_iter=100, random_state=None,
                 early_stopping=True, eps=1.0e-5, raise_on_increase=True,
                 cache_dir='.cache', nb_fit_try=1, n_jobs=1, verbose=0):
        # model hyperparameters
        self.t_r = t_r
        self.n_atoms = n_atoms
        self.hrf_model = hrf_model
        self.deactivate_v_learning = deactivate_v_learning
        self.n_times_atom = n_times_atom
        self.lbda_strategy = lbda_strategy
        self.lbda = lbda
        self.u_init_type = u_init_type
        self.prox_u = prox_u

        # convergence parameters
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.eps = eps
        self.raise_on_increase = raise_on_increase

        # technical parameters
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.cache_dir = cache_dir
        self.nb_fit_try = nb_fit_try

        # HRF atlas
        basc_atlas = fetch_atlas_basc_2015(hrf_atlas)
        self.mask_full_brain, self.atlas_rois = basc_atlas
        self.hrf_rois = dict()

        # fMRI masker
        self.masker_ = input_data.NiftiMasker(
                            mask_img=self.mask_full_brain, t_r=self.t_r,
                            memory=self.cache_dir, memory_level=1, verbose=0)

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

    def fit(self, X):
        """ Perform the Sparse Low-Rank Deconvolution Analysis (SLRDA) by
        fitting the model.

        Parameters
        ----------
        X : str, the filename of the fMRI data

        Throws
        ------
        CostFunctionIncreased : if the cost-function increases during an
            iteration, of the analysis. This can be due to the fact that the
            temporal regularization parameter is set to high
        """
        if not isinstance(X, str):
            raise ValueError("fMRI data should passed as a "
                             "filename (string), got {}".format(type(X)))

        # load the fMRI data into a matrix
        X = self.masker_.fit_transform(X).T

        # transformation of the atlas format
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
                      hrf_model=self.hrf_model,
                      deactivate_v_learning=self.deactivate_v_learning,
                      n_atoms=self.n_atoms, n_times_atom=self.n_times_atom,
                      lbda_strategy=self.lbda_strategy, lbda=self.lbda,
                      u_init_type=self.u_init_type, prox_u=self.prox_u,
                      max_iter=self.max_iter, get_obj=1, get_time=1,
                      random_seed=self.random_state,
                      early_stopping=self.early_stopping, eps=self.eps,
                      raise_on_increase=self.raise_on_increase,
                      verbose=self.verbose, n_jobs=self.n_jobs,
                      nb_fit_try=self.nb_fit_try)

        if isinstance(self.cache_dir, str):
            decompose = Memory(self.cache_dir).cache(
                                        multi_runs_learn_u_z_v_multi)
        else:
            decompose = multi_runs_learn_u_z_v_multi

        # SLRDA decomposition
        res = decompose(**params)

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
        """ Perform the Sparse Low-Rank Deconvolution Analysis (SLRDA) by
        fitting the model (same as fit).

        Parameters
        ----------
        X : str, the filename of the fMRI data

        Throws
        ------
        CostFunctionIncreased : if the cost-function increases during an
            iteration, of the analysis. This can be due to the fact that the
            temporal regularization parameter is set to high
        """
        self.fit(X)
        return self

    def transform(self, X):
        """ Perform the Sparse Low-Rank Deconvolution Analysis (SLRDA) by
        fitting the model (same as fit).

        Parameters
        ----------
        X : str, the filename of the fMRI data

        Throws
        ------
        CostFunctionIncreased : if the cost-function increases during an
            iteration, of the analysis. This can be due to the fact that the
            temporal regularization parameter is set to high
        """
        self.fit(X)
        return self

    def _check_fitted(self):
        """ Private helper, check if the Sparse Low-Rank Deconvolution Analysis
        (SLRDA) have been done.
        """
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
