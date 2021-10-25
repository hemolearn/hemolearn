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
from nilearn import datasets
import numpy as np

from hemolearn import SLRDA, plotting


###############################################################################
# Create plotting directory
# -------------------------
plot_dir = 'plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

###############################################################################
# Fetch fMRI subjects
# -------------------------
seed, TR, n_subjects = 0, 2.0, 4
adhd_dataset = datasets.fetch_adhd(n_subjects=n_subjects)
func_fnames = adhd_dataset.func[:n_subjects]
confound_fnames = adhd_dataset.confounds[:n_subjects]

###############################################################################
# Distangle the neurovascular coupling from the neural activation
# ---------------------------------------------------------------
slrda = SLRDA(n_atoms=10, t_r=TR, n_times_atom=30, lbda=0.1, max_iter=50,
              standardize=True, shared_spatial_maps=True, random_state=seed,
              verbose=1)
a_hat_img = slrda.fit_transform(func_fnames, confound_fnames=confound_fnames)

###############################################################################
# Plot the spatial maps
# ---------------------
filename = os.path.join(plot_dir, f'spatial_maps.png')
plotting.plot_spatial_maps(slrda.u_hat_img, filename=filename,
                           perc_voxels_to_retain='1%', verbose=True)

###############################################################################
# Plot the temporal activations
# -----------------------------
for n in range(n_subjects):
    filename = os.path.join(plot_dir, f'activations_{n}.png')
    plotting.plot_temporal_activations(slrda.z_hat[n], TR, filename=filename,
                                       verbose=True)

###############################################################################
# Plot vascular maps
# ------------------
vmax = np.max([np.max(slrda.a_hat[n]) for n in range(n_subjects)])
for n in range(n_subjects):
    filename = os.path.join(plot_dir, f'vascular_maps_{n}.png')
    plotting.plot_vascular_map(a_hat_img[n], display_mode='z',
                               cut_coords=np.linspace(-30, 60, 5),
                               filename=filename, vmax=vmax, verbose=True)
