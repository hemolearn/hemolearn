""" Example to recover the different spontanious tasks involved in the BOLD
signal on the ADHD"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import time
import shutil
from datetime import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import input_data, datasets, plotting
from sklearn.decomposition import FastICA


###############################################################################
# define data
TR = 2.0
adhd_dataset = datasets.fetch_adhd(n_subjects=1)
fmri_filename = adhd_dataset.func[0]
confound_filename = adhd_dataset.confounds[0]
masker = input_data.NiftiMasker(t_r=TR,
                                smoothing_fwhm=6,
                                detrend=True,
                                standardize=True,
                                low_pass=0.1,
                                high_pass=0.01,
                                memory='.cache', memory_level=1, verbose=0)
X = masker.fit_transform(fmri_filename, confounds=[confound_filename]).T
print("Data loaded shape: {} scans {} voxels".format(*X.shape))

###############################################################################
# estimation of u an z
random_seed = 1
n_atoms = 8

ica = FastICA(n_components=n_atoms)

t0 = time.time()
ica.fit(X.T)
delta_t = time.strftime("%M min %S s", time.gmtime(time.time() - t0))
print("Fitting done in {}".format(delta_t))

u_hat = ica.components_

###############################################################################
# results management
date = datetime.now()
dirname = 'results_ica_ahdh_#{0}{1}{2}{3}{4}{5}'.format(date.year, date.month,
                                                        date.day, date.hour,
                                                        date.minute,
                                                        date.second)

if not os.path.exists(dirname):
    os.makedirs(dirname)

print("archiving '{0}' under '{1}'".format(__file__, dirname))
shutil.copyfile(__file__, os.path.join(dirname, __file__))

###############################################################################
# archiving results
res = dict(u_hat=u_hat)
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
    last_retained_voxel_idx = int(0.05 * u_k.shape[0])  # retain 5% voxels
    th = np.sort(u_k)[-last_retained_voxel_idx]
    img_u_k = masker.inverse_transform(u_k)
    img_u.append(img_u_k)
    plotting.plot_stat_map(img_u_k, title="Map-{}".format(k), colorbar=True,
                           threshold=th)
    img_u_k.to_filename(os.path.join(dirname, "u_{0:03d}.nii".format(k)))
    plt.savefig(os.path.join(dirname, "u_{0:03d}.pdf".format(k)), dpi=150)
plotting.plot_prob_atlas(img_u, title='All spatial maps')
filename = os.path.join(dirname, "all_U.pdf")
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename, dpi=150)

import subprocess  # XXX hack to concatenate the spatial maps in one pdf
pdf_files = os.path.join(dirname, 'u_*.pdf')
pdf_file = os.path.join(dirname, 'U.pdf')
subprocess.call("pdftk {} cat output {}".format(pdf_files, pdf_file),
                shell=True)
subprocess.call("rm -f {}".format(pdf_files), shell=True)
print("Saving plot under '{0}'".format(pdf_file))