""" HemoLearn is a Python module to estimate the Haemodynamic Response Function
(HRF) in brain from resting-state or task fMRI data (BOLD signal). It relies on
a Sparse Low-Rank Deconvolution Analysis (SLRDA) to distangles the
neurovascular coupling from the the neural activity.

The advantages of SLRDA are:

* The estimation of the HRF for each regions of the given vascular atlas.
* The decomposition of the neural activity in a set of temporal components and
its associated spatial map that describe a function network in the brain.

The disadvantages of SLRDA include:

* If the temporal resolution in the fMRI data is too low (TR > 1s) it's likely
that the analysis will not found major difference between the HRFs. This is due
to the fact that the common time-to-peak difference between HRFs within the
same brain varies at maximum of 1 s, if the temporal resolution is greater that
this, no effect will be estimated.
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from nilearn import input_data, image, signal

from .deconvolution import multi_runs_blind_deconvolution_multiple_subjects
from .atlas import (fetch_harvard_vascular_atlas, fetch_basc_vascular_atlas,
                    fetch_aal3_vascular_atlas, split_atlas)
from .build_numba import build_numba_functions_of_hemolearn
from .utils import sort_by_expl_var


# call each functions defined with Numba to build the functions
build_numba_functions_of_hemolearn()


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
    shared_spatial_maps : bool, whether or not to learn a single set of
        spatials maps accross subjects.
    hrf_model : str, (default='3_basis_hrf'), type of HRF model, possible
        choice are ['3_basis_hrf', '2_basis_hrf', 'scaled_hrf']
    hrf_atlas : str, func, or None, (default='aal3'), atlas type, possible
        choice are ['harvard', 'basc', given-function]. None default option will
        lead to fetch the Harvard-Oxford parcellation.
    atlas_kwargs : dict, (default=dict()), additional kwargs for the atlas,
        if a function is passed.
    n_scales : str, (default='scale122'), select the number of scale if
        hrf_atlas == 'basc'.
    deactivate_v_learning : bool, (default=False), option to force the
        estimated HRF to its initial value.
    deactivate_z_learning : bool, (default=False), option to force the
        estimated z to its initial value.
    prox_z : str, (default='tv'), temporal proximal operator should be in
        ['tv', 'l1', 'l2', 'elastic-net']
    lbda_strategy str, (default='ratio'), strategy to fix the temporal
        regularization parameter, possible choice are ['ratio', 'fixed']
    lbda : float, (default=0.1), whether the temporal regularization parameter
        if lbda_strategy == 'fixed' or the ratio w.r.t lambda max if
        lbda_strategy == 'ratio'
    rho : float, (default=2.0), the elastic-net temporal regularization
        parameter
    delta_init : float, (default=1.0), the initialization value for the HRF
        dilation parameter
    u_init_type : str, (default='ica'), strategy to init u, possible value are
        ['gaussian_noise', 'ica', 'patch']
    z_init : None or array, (default=None), initialization of z, if None, z is
        initialized to zero
    prox_u : str, (default='l2-positive-ball'), constraint to impose on the
        spatial maps possible choice are ['l2-positive-ball',
        'l1-positive-simplex', 'positive']
    standardize : bool, (default=False), if standardize is True, the
        time-series are centered and normed: their mean is put to 0 and their
        variance to 1 in the time dimension.
    detrend : bool, (defaul==False), whether or not to detrend the BOLD signal
    low_pass : float or None, (defaul==None), the limit low freq pass band to
        clean the BOLD signal
    high_pass : float or None, (default=None), the limit high freq pass band to
        clean the BOLD signal
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
                 hrf_atlas='aal3', atlas_kwargs=dict(), n_scales='scale122',
                 prox_z='tv', lbda_strategy='ratio', lbda=0.1, rho=2.0,
                 delta_init=1.0, u_init_type='ica', eta=10.0, z_init=None,
                 prox_u='l1-positive-simplex', deactivate_v_learning=False,
                 shared_spatial_maps=False, deactivate_z_learning=False,
                 standardize=False, detrend=False, low_pass=None,
                 high_pass=None, max_iter=100, random_state=None,
                 early_stopping=True, eps=1.0e-5, raise_on_increase=True,
                 cache_dir='__cache__', nb_fit_try=1, n_jobs=1, verbose=0):

        # model hyperparameters
        self.t_r = t_r
        self.n_atoms = n_atoms
        self.hrf_model = hrf_model
        self.hrf_atlas = hrf_atlas
        self.n_scales = n_scales
        self.shared_spatial_maps = shared_spatial_maps
        self.deactivate_v_learning = deactivate_v_learning
        self.deactivate_z_learning = deactivate_z_learning
        self.n_times_atom = n_times_atom
        self.prox_z = prox_z
        self.lbda_strategy = lbda_strategy
        self.lbda = lbda
        self.rho = rho
        self.delta_init = delta_init
        self.u_init_type = u_init_type
        self.eta = eta
        self.z_init = z_init
        self.prox_u = prox_u

        # convergence parameters
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.eps = eps
        self.raise_on_increase = raise_on_increase

        # preprocessing parameters
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass

        # technical parameters
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.cache_dir = cache_dir
        self.nb_fit_try = nb_fit_try

        # HRF atlas
        if self.hrf_atlas == 'aal3':
            self.mask_full_brain, self.atlas_rois = fetch_aal3_vascular_atlas()
        elif self.hrf_atlas == 'harvard':
            res = fetch_harvard_vascular_atlas()
            self.mask_full_brain, self.atlas_rois = res
        elif self.hrf_atlas == 'basc':
            n_scales_ = f"scale{int(n_scales)}"
            res = fetch_atlas_basc_2015(n_scales=n_scales_)
            self.mask_full_brain, self.atlas_rois = res
        elif hasattr(self.hrf_atlas, '__call__'):
            res = self.hrf_atlas(**atlas_kwargs)
            self.mask_full_brain, self.atlas_rois = res
        else:
            raise ValueError(f"hrf_atlas should belong to ['aal3', 'harvard', "
                             f"'basc', given-function], got {self.hrf_atlas}")
        self.hrf_rois = dict()

        # fMRI masker
        self.masker_ = input_data.NiftiMasker(
                            mask_img=self.mask_full_brain, t_r=self.t_r,
                            memory=self.cache_dir, memory_level=1, verbose=0)

        # model parameters
        self.n_subjects = None
        self.z_hat_ = None
        self.u_hat_ = None
        self.a_hat_ = None
        self.u_hat_img_ = None
        self.a_hat_img_ = None

        # derivated model parameters
        self.Dz_hat_ = None
        self.v_hat_ = None

        # auxilaries parameters
        self.roi_label_from_hrf_idx = None
        self.v_init_ = None
        self.lbda_ = lbda
        self.lobj_ = None
        self.ltime_ = None

    def fit(self, X_fnames, confound_fnames=None):
        """ Perform the Sparse Low-Rank Deconvolution Analysis (SLRDA) by
        fitting the model.

        Parameters
        ----------
        X_fnames : str, the filename of the fMRI data
        confound_fnames : Nifti-like img or None, (default=None), list of
            confounds (2D arrays or filenames pointing to CSV files). Must be
            of same length than imgs_list.

        Throws
        ------
        CostFunctionIncreased : if the cost-function increases during an
            iteration, of the analysis. This can be due to the fact that the
            temporal regularization parameter is set to high
        """
        if isinstance(X_fnames, str):
            X_fnames = [X_fnames]

        if isinstance(X_fnames, list):
            for x_i in X_fnames:
                if not isinstance(x_i, str):
                    raise ValueError(f"fMRI data should passed as a "
                                     f"filename (string), got {type(x_i)}")

        else:
            raise ValueError(f"fMRI data should passed as a list of"
                             f"filenames (string), got {type(X_fnames)}")

        self.n_subjects = len(X_fnames)

        # load the fMRI data into a matrix, given-X being a filename,
        # produced-X being a 2d-array (n_voxels, n_time)
        self.masker_.fit(X_fnames)
        X = [self.masker_.transform_single_imgs(x_fname).T
             for x_fname in X_fnames]

        # transformation of the atlas format:
        # flattent the HRF ROIs
        rois = self.masker_.transform(self.atlas_rois).astype(int).ravel()
        # list the indices for each ROIs (already sorted)
        index = np.arange(rois.shape[-1])
        # gather for each ROIs (defined by its index) the concerned voxels idx
        for roi_label in np.unique(rois):
            self.hrf_rois[roi_label] = index[roi_label == rois]
        _, rois_label, _ = split_atlas(self.hrf_rois)
        # roi_label_from_hrf_idx: split the HRF atlas into a table of indices
        # for each ROIs, a vector of labels for each ROIs and the number of
        # ROIs from a dict atlas
        self.roi_label_from_hrf_idx = rois_label
        self.n_hrf_rois = len(rois_label)

        # preprocess the input data
        X_clean = []
        for n in range(self.n_subjects):

            if confound_fnames is not None:
                # XXX confounds should be CSV file with '\t' sep
                confounds = pd.read_csv(confound_fnames[n], sep='\t')
            else:
                confounds = None

            if self.verbose > 0:
                print(f"[SLRDA] Clean subject '{X_fnames[n]}'")

            x_clean = signal.clean(X[n].T, runs=None, detrend=self.detrend,
                                   standardize=self.standardize,
                                   confounds=confounds, low_pass=self.low_pass,
                                   high_pass=self.high_pass, t_r=self.t_r,
                                   ensure_finite=True)

            X_clean.append(x_clean.T)

        if self.verbose > 0:
            if self.n_jobs > 1:
                print(f"[SLRDA] Running {self.nb_fit_try} fit(s) on "
                      f"{self.n_subjects} subject(s) in parallel on "
                      f"{self.n_jobs} CPU")
            else:
                print(f"[SLRDA] Running {self.nb_fit_try} fit(s) on "
                      f"{self.n_subjects} subject(s) in series")

        params = dict(
                X=X_clean, t_r=self.t_r, hrf_rois=self.hrf_rois,
                hrf_model=self.hrf_model,
                shared_spatial_maps=self.shared_spatial_maps,
                deactivate_v_learning=self.deactivate_v_learning,
                deactivate_z_learning=self.deactivate_z_learning,
                n_atoms=self.n_atoms, n_times_atom=self.n_times_atom,
                prox_z=self.prox_z, lbda_strategy=self.lbda_strategy,
                lbda=self.lbda, rho=self.rho, delta_init=self.delta_init,
                u_init_type=self.u_init_type, eta=self.eta, z_init=self.z_init,
                prox_u=self.prox_u, max_iter=self.max_iter,
                random_seed=self.random_state,
                early_stopping=self.early_stopping, eps=self.eps,
                raise_on_increase=self.raise_on_increase,
                verbose=self.verbose, n_jobs=self.n_jobs,
                nb_fit_try=self.nb_fit_try)

        # handle the caching option of the decomposition
        if isinstance(self.cache_dir, str):
            decompose = Memory(location=self.cache_dir, verbose=0).cache(
                            multi_runs_blind_deconvolution_multiple_subjects)
        else:
            decompose = multi_runs_blind_deconvolution_multiple_subjects

        # SLRDA decomposition
        res = decompose(**params)

        self.z_hat_ = res[0]
        self.Dz_hat_ = res[1]
        self.u_hat_ = res[2]
        self.a_hat_ = res[3]
        self.v_hat_ = res[4]

        if not self.shared_spatial_maps:

            for n in range(self.n_subjects):

                u_hat, z_hat, _ = sort_by_expl_var(self.u_hat_[n],
                                                   self.z_hat_[n],
                                                   self.v_hat_[n],
                                                   self.hrf_rois)

                self.u_hat_[n] = u_hat
                self.z_hat_[n] = z_hat

        self.v_init_ = res[5]
        self.lbda_ = res[6]
        self.lobj_ = res[7]
        self.ltime_ = res[8]

        if self.hrf_model == 'scaled_hrf':
            a_hat_img_ = []
            for n in range(self.n_subjects):
                raw_atlas_rois = np.copy(self.atlas_rois.get_fdata())
                for m in range(self.n_hrf_rois):
                    label = int(self.roi_label_from_hrf_idx[m])
                    raw_atlas_rois[raw_atlas_rois == label] = self.a_hat_[n][m]
                a_hat_img_.append(image.new_img_like(self.atlas_rois,
                                                     raw_atlas_rois))
            self.a_hat_img_ = a_hat_img_

        else:
            self.a_hat_img_ = None

        n_spatial_maps = 1 if self.shared_spatial_maps else self.n_subjects

        u_hat_img_ = []
        for n in range(n_spatial_maps):
            u_hat_n_img_ = []
            for k in range(self.n_atoms):
                u_hat_nk_ = self.u_hat_[n][k]
                u_hat_nk_img_ = self.masker_.inverse_transform(u_hat_nk_)
                u_hat_n_img_.append(u_hat_nk_img_)
            u_hat_img_.append(u_hat_n_img_)
        self.u_hat_img_ = u_hat_img_

        return self

    def fit_transform(self, X_fnames, confound_fnames=None):
        """ Perform the Sparse Low-Rank Deconvolution Analysis (SLRDA) by
        fitting the model (same as fit).

        Parameters
        ----------
        X_fnames : str, the filename of the fMRI data

        Throws
        ------
        CostFunctionIncreased : if the cost-function increases during an
            iteration, of the analysis. This can be due to the fact that the
            temporal regularization parameter is set to high
        """
        self.fit(X_fnames, confound_fnames=confound_fnames)
        return self.a_hat_img

    def transform(self, X_fnames, confound_fnames=None):
        """ Perform the Sparse Low-Rank Deconvolution Analysis (SLRDA) by
        fitting the model (same as fit).

        Parameters
        ----------
        X_fnames : str, the filename of the fMRI data

        Throws
        ------
        CostFunctionIncreased : if the cost-function increases during an
            iteration, of the analysis. This can be due to the fact that the
            temporal regularization parameter is set to high
        """
        self.self._check_fitted()
        return self.a_hat_img

    def _check_fitted(self):
        """ Private helper, check if the Sparse Low-Rank Deconvolution Analysis
        (SLRDA) have been done.
        """
        if self.n_subjects is None:
            raise NotFittedError("SLDRA must be fitted first.")

    @property
    def u_hat(self):
        self._check_fitted()
        if self.n_subjects == 1 or self.shared_spatial_maps:
            return self.u_hat_[0]
        return self.u_hat_

    @property
    def u_hat_img(self):
        self._check_fitted()
        if self.n_subjects == 1 or self.shared_spatial_maps:
            return self.u_hat_img_[0]
        return self.u_hat_img_

    @property
    def z_hat(self):
        self._check_fitted()
        if self.n_subjects == 1:
            return self.z_hat_[0]
        return self.z_hat_

    @property
    def Dz_hat(self):
        self._check_fitted()
        if self.n_subjects == 1:
            return self.Dz_hat_[0]
        return self.Dz_hat_

    @property
    def a_hat(self):
        self._check_fitted()
        if self.n_subjects == 1:
            return self.a_hat_[0]
        return self.a_hat_

    @property
    def a_hat_img(self):
        self._check_fitted()
        if self.n_subjects == 1:
            return self.a_hat_img_[0]
        return self.a_hat_img_

    @property
    def v_hat(self):
        self._check_fitted()
        if self.n_subjects == 1:
            return self.v_hat_[0]
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
