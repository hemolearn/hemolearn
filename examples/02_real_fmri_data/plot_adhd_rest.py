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
from nilearn import datasets
import numpy as np

from hemolearn import SLRDA
from hemolearn.plotting import (plot_spatial_maps, plot_temporal_activations,
                                plot_vascular_map)


t0_total = time.time()

###############################################################################
# Create plotting directory
# -------------------------
plot_dir = 'plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

###############################################################################
# Fetch fMRI subjects
seed = 0
TR = 2.0
n_subjects = 4
adhd_dataset = datasets.fetch_adhd(n_subjects=n_subjects)
func_fnames = adhd_dataset.func[:n_subjects]
confound_fnames = adhd_dataset.confounds[:n_subjects]

###############################################################################
# Distangle the neurovascular coupling from the neural activation
# ---------------------------------------------------------------
slrda = SLRDA(n_atoms=10, t_r=TR, n_times_atom=20, lbda=0.75, max_iter=30,
              eps=1.0e-3, shared_spatial_maps=True, random_state=seed,
              verbose=2)

t0 = time.time()
slrda.fit(func_fnames, confound_fnames)
delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
print(f"Fitting done in {delta_t}")

###############################################################################
# Plot the spatial maps
# ---------------------
if slrda.shared_spatial_maps or n_subjects == 1:
    filename = os.path.join(plot_dir, f'spatial_maps.png')
    plot_spatial_maps(slrda.u_hat_img, filename=filename,
                      perc_voxels_to_retain='10%', verbose=True)
else:
    for n in range(n_subjects):
        filename = os.path.join(plot_dir, f'spatial_maps_{n}.png')
        plot_spatial_maps(slrda.u_hat_img[n], filename=filename,
                          perc_voxels_to_retain='10%', verbose=True)

###############################################################################
# Plot the temporal activations
# -----------------------------
if n_subjects == 1:
    filename = os.path.join(plot_dir, f'activations.png')
    plot_temporal_activations(slrda.z_hat, TR, filename=filename, verbose=True)
else:
    for n in range(n_subjects):
        filename = os.path.join(plot_dir, f'activations_{n}.png')
        plot_temporal_activations(slrda.z_hat[n], TR, filename=filename,
                                  verbose=True)

###############################################################################
# Plot vascular maps
# ------------------
if n_subjects == 1:
    filename = os.path.join(plot_dir, f'vascular_maps.png')
    plot_vascular_map(slrda.a_hat_img, display_mode='z',
                      cut_coords=np.linspace(-30, 60, 5),
                      filename=filename, verbose=True)
else:
    for n in range(n_subjects):
        filename = os.path.join(plot_dir, f'vascular_maps_{n}.png')
        plot_vascular_map(slrda.a_hat_img[n], display_mode='z',
                          cut_coords=np.linspace(-30, 60, 5),
                          filename=filename, verbose=True)

###############################################################################
# Display the runtime of the script
# ---------------------------------
delta_t = time.gmtime(time.time() - t0_total)
delta_t = time.strftime("%H h %M min %S s", delta_t)
print(f"Script runs in {delta_t}")
