"""
EUSIPCO 2019 CHerkaoui et a.l. experimental reproduction
========================================================

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
import pickle
from nilearn import datasets

from hemolearn import SLRDA
from hemolearn.utils import (fmri_preprocess, get_unique_dirname,
                             sort_atoms_by_explained_variances)
from hemolearn.plotting import (plotting_spatial_comp, plotting_temporal_comp,
                                plotting_obj_values)


# %%

dirname = get_unique_dirname("results_slrda_adhd_#")
if not os.path.exists(dirname):
    os.makedirs(dirname)

# %%

TR = 2.0
adhd_dataset = datasets.fetch_adhd(n_subjects=1)
func_fname = adhd_dataset.func[0]
confound_fname = adhd_dataset.confounds[0]
X, _, _, _ = = fmri_preprocess(func_fname, smoothing_fwhm=6.0,
                               standardize=True, detrend=True, low_pass=0.1,
                               high_pass=0.01, t_r=TR, memory='.cache',
                               verbose=0, confounds=confound_fname)
seed = None
n_atoms = 20
hrf_atlas = 'scale007'

# %%

slrda = SLRDA(n_atoms=n_atoms, t_r=TR, hrf_atlas=hrf_atlas, n_times_atom=20,
              hrf_model='scaled_hrf', lbda=0.6, max_iter=100, eps=1.0e-3,
              deactivate_v_learning=True, prox_u='l1-positive-simplex',
              raise_on_increase=False, random_state=seed, n_jobs=1,
              nb_fit_try=1, verbose=2)

t0 = time.time()
slrda.fit(X)
delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
print("Fitting done in {}".format(delta_t))

# %%

hrf_ref, roi_label_from_hrf_idx = slrda.v_init, slrda.roi_label_from_hrf_idx
pobj, times, a_hat, v_hat = slrda.lobj, slrda.ltime, slrda.a_hat, slrda.v_hat
u_hat, z_hat, variances = sort_atoms_by_explained_variances(slrda.u_hat,
                                                            slrda.z_hat,
                                                            slrda.v_hat,
                                                            slrda.hrf_rois)

res = dict(pobj=pobj, times=times, u_hat=u_hat, v_hat=v_hat, z_hat=z_hat)
filename = os.path.join(dirname, "results.pkl")
print("Pickling results under '{0}'".format(filename))
with open(filename, "wb") as pfile:
    pickle.dump(res, pfile)

# %%

plotting_spatial_comp(u_hat, variances, slrda.masker_, plot_dir=dirname,
                      perc_voxels_to_retain=0.1, verbose=True)
plotting_temporal_comp(z_hat, variances, TR, plot_dir=dirname, verbose=True)
plotting_obj_values(times, pobj, plot_dir=dirname, verbose=True)
