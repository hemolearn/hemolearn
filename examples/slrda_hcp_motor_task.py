""" Example to recover the different neural temporal activities, the associated
functional networks maps and the HRFs per ROIs in the fMRI data, on the HCP
dataset motor task. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
is_travis = ('TRAVIS' in os.environ)
if is_travis:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import shutil
import pickle
import matplotlib.pyplot as plt
from nilearn import datasets

from seven import SLRDA
from seven.utils import (fmri_preprocess, sort_atoms_by_explained_variances,
                         get_unique_dirname)
from seven.plotting import (plotting_spatial_comp, plotting_temporal_comp,
                            plotting_obj_values, plotting_hrf,
                            plotting_hrf_stats)

from _utils import (fetch_subject_list, _get_hcp_motor_task_fmri_fname,
                    get_paradigm_hcp_motor_task)
from _utils import TR_HCP_MOTOR as TR


dirname = get_unique_dirname("results_slrda_hcp_motor_task_#")
if not os.path.exists(dirname):
    os.makedirs(dirname)

print("archiving '{0}' under '{1}'".format(__file__, dirname))
shutil.copyfile(__file__, os.path.join(dirname, __file__))

subject_id = fetch_subject_list()[0]
func_fname, anat_fname = _get_hcp_motor_task_fmri_fname(subject_id,
                                                        anat_data=True)
X = fmri_preprocess(func_fname, smoothing_fwhm=6.0, standardize=True,
                    detrend=True, low_pass=0.1, high_pass=0.01, t_r=TR,
                    memory='.cache', verbose=0)
seed = None
n_atoms = 10
hrf_atlas = 'scale036'
slrda = SLRDA(n_atoms=n_atoms, t_r=TR, hrf_atlas=hrf_atlas, n_times_atom=60,
              hrf_model='3_basis_hrf', lbda=1.0e-2, max_iter=100,
              raise_on_increase=True, random_state=seed, n_jobs=1,
              nb_fit_try=1, verbose=2)

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
                      verbose=True)
d = 12.0 * TR
l_onset_t = []
conditions = ['lh', 'rh', 'lf', 'rf']
paradigm_full, _ = get_paradigm_hcp_motor_task(subject_id)
for cond in conditions:
    paradigm = paradigm_full[paradigm_full['trial_type'] == cond]
    mask_cond = (paradigm['trial_type'] == cond).values
    l_onset_t.extend(list(paradigm['onset'][mask_cond].values / TR))
def plot_paradigm():
    for t in l_onset_t:
        plt.axvspan(t, t+d, facecolor='gray', alpha=0.2)
plotting_temporal_comp(z_hat, variances, TR, plot_dir=dirname,
                       aux_plots=plot_paradigm, verbose=True)
plotting_obj_values(times, pobj, plot_dir=dirname, verbose=True)
plotting_hrf(v_hat, TR, hrf_atlas, roi_label_from_hrf_idx,
             hrf_ref=hrf_ref, normalized=True, plot_dir=dirname, verbose=True)
plotting_hrf_stats(v_hat, TR, hrf_atlas, roi_label_from_hrf_idx,
                   hrf_ref=None, stat_type='tp', plot_dir=dirname,
                   verbose=True)
plotting_hrf_stats(v_hat, TR, hrf_atlas, roi_label_from_hrf_idx,
                   hrf_ref=None, stat_type='fwhm', plot_dir=dirname,
                   verbose=True)
