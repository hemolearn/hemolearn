""" Example to recover the different neural temporal activities, the associated
functional networks maps and the HRFs per ROIs in the fMRI data, on the HCP
dataset resting-state. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
is_travis = ('TRAVIS' in os.environ)
if is_travis:
    import matplotlib
    matplotlib.use('Agg')

import time
import shutil
import pickle

from seven import SLRDA
from seven.hrf_model import spm_hrf
from seven.utils import (fmri_preprocess, sort_atoms_by_explained_variances,
                         get_unique_dirname)
from seven.plotting import (plotting_spatial_comp, plotting_temporal_comp,
                            plotting_obj_values, plotting_hrf,
                            plotting_hrf_stats)

from utils import fetch_subject_list, _get_hcp_rest_fmri_fname
from utils import TR_HCP_REST as TR


dirname = get_unique_dirname("results_slrda_hcp_rest_#")
if not os.path.exists(dirname):
    os.makedirs(dirname)

print("archiving '{0}' under '{1}'".format(__file__, dirname))
shutil.copyfile(__file__, os.path.join(dirname, __file__))

subject_id = fetch_subject_list()[0]
func_fname, anat_fname = _get_hcp_rest_fmri_fname(subject_id, anat_data=True)
X = fmri_preprocess(func_fname, smoothing_fwhm=6.0, standardize=True,
                    detrend=True, low_pass=0.1, high_pass=0.01, t_r=TR,
                    memory='.cache', verbose=0)
seed = None
n_atoms = 30
hrf_atlas = 'scale122'
slrda = SLRDA(n_atoms=n_atoms, t_r=TR, hrf_atlas=hrf_atlas, n_times_atom=60,
              hrf_model='scaled_hrf', lbda=1.0e-3, max_iter=150,
              deactivate_v_learning=False, prox_u='l1-positive-simplex',
              raise_on_increase=False, random_state=seed, n_jobs=3,
              nb_fit_try=3, verbose=2)

t0 = time.time()
slrda.fit(X)
delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
print("Fitting done in {}".format(delta_t))

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

plotting_spatial_comp(u_hat, variances, slrda.masker_, plot_dir=dirname,
                      perc_voxels_to_retain=0.1, bg_img=anat_fname,
                      display_mode='z', verbose=True)
plotting_temporal_comp(z_hat, variances, TR, plot_dir=dirname, verbose=True)
plotting_obj_values(times, pobj, plot_dir=dirname, verbose=True)
plotting_hrf(v_hat, TR, hrf_atlas, roi_label_from_hrf_idx,
             hrf_ref=hrf_ref, normalized=True, plot_dir=dirname, verbose=True)
hrf_ref = spm_hrf(TR, 60)
plotting_hrf_stats(v_hat, TR, hrf_atlas, roi_label_from_hrf_idx,
                   hrf_ref=hrf_ref, stat_type='tp', plot_dir=dirname,
                   verbose=True)
plotting_hrf_stats(v_hat, TR, hrf_atlas, roi_label_from_hrf_idx,
                   hrf_ref=hrf_ref, stat_type='fwhm', plot_dir=dirname,
                   verbose=True)