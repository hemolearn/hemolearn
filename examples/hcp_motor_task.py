""" Example to recover the different spontanious tasks involved in the BOLD
signal on the HCP dataset"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import time
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import input_data, datasets, image, plotting

from alphacsc.utils.hrf_model import spm_hrf

from seven import SLRDM
from seven.utils import sort_atoms_by_explained_variances, get_unique_dirname
from seven.hrf_model import spm_hrf

from utils import fetch_subject_list, get_hcp_fmri_fname, TR_HCP
from utils import get_protocol_hcp


###############################################################################
# define data
subject_id = fetch_subject_list()[0]
func_fname, anat_fname = get_hcp_fmri_fname(subject_id, anat_data=True)

###############################################################################
# estimation of u an z
random_seed = 0
n_atoms = 8
n_times_atom = 30
v = spm_hrf(TR_HCP, n_times_atom)

cdl = SLRDM(n_atoms=n_atoms, v=v, t_r=TR_HCP, lbda=1.0e-1, max_iter=50,
            random_state=random_seed, n_jobs=3, verbose=1)

t0 = time.time()
cdl.fit(func_fname, nb_fit_try=3)
delta_t = time.strftime("%M min %S s", time.gmtime(time.time() - t0))

print("Fitting done in {}".format(delta_t))

# gather and sort activations and spatial components
pobj, times = cdl.lobj, cdl.ltime
u_hat, Lz_hat, variances = sort_atoms_by_explained_variances(cdl.u_hat,
                                                             cdl.Lz_hat, v)

###############################################################################
# results management
dirname = get_unique_dirname("results_hcp_#")

if not os.path.exists(dirname):
    os.makedirs(dirname)

print("archiving '{0}' under '{1}'".format(__file__, dirname))
shutil.copyfile(__file__, os.path.join(dirname, __file__))

###############################################################################
# archiving results
res = dict(pobj=pobj, times=times, u_hat=u_hat, Lz_hat=Lz_hat)
filename = os.path.join(dirname, "results.pkl")
print("Pickling results under '{0}'".format(filename))
with open(filename, "wb") as pfile:
    pickle.dump(res, pfile)

###############################################################################
# plotting
# u
img_u =[]
for k in range(1, n_atoms + 1):
    u_k = u_hat[k - 1]
    last_retained_voxel_idx = int(0.1 * u_k.shape[0])  # retain 10% voxels
    th = np.sort(u_k)[-last_retained_voxel_idx]
    expl_var = variances[k - 1]
    title = "Map-{} (explained variance = {:.2e})".format(k, expl_var)
    img_u_k = cdl.masker_.inverse_transform(u_k)
    img_u.append(img_u_k)
    plotting.plot_stat_map(img_u_k, bg_img=anat_fname, title=title,
                           colorbar=True, threshold=th)
    img_u_k.to_filename(os.path.join(dirname, "u_{0:03d}.nii".format(k)))
    plt.savefig(os.path.join(dirname, "u_{0:03d}.pdf".format(k)), dpi=150)
plotting.plot_prob_atlas(img_u, title='All spatial maps')
filename = os.path.join(dirname, "all_U.pdf")
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename, dpi=150)

# Lz
plt.figure("Temporal atoms", figsize=(8, 5 * n_atoms))
conditions = ['lh', 'rh', 'lf', 'rf']
all_pe = []
for i, cond in enumerate(conditions):
    _, _, pe, _ = get_protocol_hcp(subject_id, cond)
    all_pe.append(pe)
all_pe = np.sum(all_pe, axis=0)
n_times_valid = Lz_hat.shape[1]
n_scans = len(all_pe)
_xticks = [0, int(n_times_valid / 2.0), int(n_times_valid), n_scans]
_xticks_labels = [
    0,
    time.strftime("%Mm%Ss", time.gmtime(int(TR_HCP * n_times_valid / 2.0))),
    time.strftime("%Mm%Ss", time.gmtime(int(TR_HCP * n_times_valid))),
    ]
for k in range(1, n_atoms + 1):
    expl_var = variances[k - 1]
    Lz_k = Lz_hat[k - 1].T
    Lz_k -= np.mean(Lz_k)
    Lz_k /= np.max(np.abs(Lz_k))
    plt.subplot(n_atoms, 1, k)
    plt.plot(Lz_k, lw=5.0)
    plt.plot(all_pe, lw=5.0, color='red')
    plt.axhline(0.0, color='black', lw=3.0)
    plt.title("Atom-{} (explained variance = {:.2e})".format(k, expl_var),
              fontsize=20)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Signal change [%]", fontsize=20)
    plt.xticks(_xticks, _xticks_labels, fontsize=20)
    plt.yticks([-1, 0, 1], fontsize=20)
    plt.grid()
plt.tight_layout()
filename = os.path.join(dirname, "Lz.pdf")
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename, dpi=150)

# pobj
t_start = 1.0e-1
min_obj = 1.0-6
plt.figure("Cost function (%)", figsize=(6, 6))
pobj -= (np.min(pobj) - min_obj)
pobj /= pobj[0]
plt.loglog(np.cumsum(times) +  t_start, pobj, lw=5.0)
plt.title("Evolution of global cost-function", fontsize=20)
plt.xlabel('Time [s]', fontsize=20)
plt.ylabel('Log normalized cost function', fontsize=20)
plt.grid()
filename = os.path.join(dirname, "pobj.pdf")
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename, dpi=150)

import subprocess  # XXX hack to concatenate the spatial maps in one pdf
pdf_files = os.path.join(dirname, 'u_*.pdf')
pdf_file = os.path.join(dirname, 'U.pdf')
subprocess.call("pdftk {} cat output {}".format(pdf_files, pdf_file),
                shell=True)
subprocess.call("rm -f {}".format(pdf_files), shell=True)
print("Saving plot under '{0}'".format(pdf_file))
