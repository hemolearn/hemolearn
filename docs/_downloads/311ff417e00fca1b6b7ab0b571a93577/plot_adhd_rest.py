"""
Real fMRI data example
======================

Example to recover the different neural temporal activities, the associated
functional networks maps and the HRFs per ROIs in the fMRI data, on the ADHD
dataset resting-state.

.. contents:: **Contents**
    :local:
    :depth: 1

"""

# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import time
import numpy as np
from nilearn import datasets

from hemolearn import SLRDA
from hemolearn.utils import fmri_preprocess, sort_atoms_by_explained_variances
from hemolearn.plotting import (plotting_spatial_comp, plotting_temporal_comp,
                                plotting_hrf_stats)


# %%

plot_dir = 'plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# %%

TR = 2.0
adhd_dataset = datasets.fetch_adhd(n_subjects=1)
func_fname = adhd_dataset.func[0]
confound_fname = adhd_dataset.confounds[0]
X, _, _, _ = fmri_preprocess(func_fname, smoothing_fwhm=6.0, standardize=True,
                             detrend=True, low_pass=0.1, high_pass=0.01, t_r=TR,
                             memory='__cache__', verbose=0,
                             confounds=confound_fname)

seed = np.random.randint(0, 1000)
print(f'Seed used = {seed}')

# %%

slrda = SLRDA(n_atoms=10, t_r=TR, n_times_atom=20,
              hrf_model='scaled_hrf', lbda=0.9, max_iter=50,
              eps=1.0e-3, deactivate_v_learning=False,
              prox_u='l1-positive-simplex', raise_on_increase=True,
              random_state=seed, n_jobs=1, cache_dir='__cache__',
              nb_fit_try=1, verbose=0)

t0 = time.time()
slrda.fit(X)
delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
print("Fitting done in {}".format(delta_t))

# %%

hrf_ref = slrda.v_init
roi_label_from_hrf_idx = slrda.roi_label_from_hrf_idx
pobj, times, a_hat = slrda.lobj, slrda.ltime, slrda.a_hat
v_hat = slrda.v_hat
u_hat, z_hat, variances = sort_atoms_by_explained_variances(slrda.u_hat,
                                                            slrda.z_hat,
                                                            slrda.v_hat,
                                                            slrda.hrf_rois)

# %%

plotting_spatial_comp(u_hat, variances, slrda.masker_,
                      plot_dir=plot_dir, perc_voxels_to_retain=0.1,
                      fname='u.png', verbose=True)

# %%

plotting_temporal_comp(z_hat, variances, TR, plot_dir=plot_dir,
                       fname='z.png', verbose=True)

# %%

plotting_hrf_stats(v=v_hat, t_r=TR, masker=slrda.masker_, hrf_ref=None,
                   stat_type='fwhm', plot_dir=plot_dir, fname='v_fwhm.png',
                   verbose=True)
