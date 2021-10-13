"""
EUSIPCO 2019 CHerkaoui et a.l. experimental reproduction
========================================================

Example to obtain the estimation error of each parameter of the model.

.. contents:: **Contents**
    :local:
    :depth: 1

"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from hemolearn.simulated_data import simulated_data
from hemolearn.deconvolution import \
                                multi_runs_blind_deconvolution_single_subject


t0_total = time.time()

###############################################################################
# Create plotting directory
# -------------------------
plot_dir = 'plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

###############################################################################
# Collect the estimation errors
# -----------------------------
mean_min_Dz_errs, std_min_Dz_errs = [], []
mean_min_u_errs, std_min_u_errs = [], []
nb_trial = 100
l_snr = [0.1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0]
for snr in l_snr:

    t0 = time.time()

    min_Dz_errs, min_u_errs = [], []
    for _ in range(nb_trial):

        TR = 1.0
        n_voxels, n_atoms, n_times_valid, n_times_atom = 100, 2, 100, 30
        noisy_X, _, u, v, z, hrf_rois = simulated_data(
                                    n_voxels=n_voxels,
                                    n_times_valid=n_times_valid,
                                    n_times_atom=n_times_atom, snr=snr)

        lbdas = np.linspace(0.1, 0.8, 20)
        u_errs, Dz_errs = [], []
        for lbda in lbdas:

            try:
                results = multi_runs_blind_deconvolution_single_subject(
                        noisy_X, t_r=TR, hrf_rois=hrf_rois, n_atoms=n_atoms,
                        deactivate_v_learning=True,
                        prox_u='l1-positive-simplex',
                        n_times_atom=n_times_atom, hrf_model='scaled_hrf',
                        lbda_strategy='ratio', lbda=lbda,
                        u_init_type='gaussian_noise', max_iter=30,
                        get_obj=True, get_time=True, raise_on_increase=False,
                        random_seed=None, n_jobs=4, nb_fit_try=4, verbose=0)

            except AssertionError as e:
                # lbda is too big...
                continue
            z_hat, _, u_hat, _, _, _, _, pobj, times = results

            # rename all variables
            u_0 = u[0, :]
            u_1 = u[1, :]
            z_0 = z[0, :]
            z_1 = z[1, :]
            u_0_hat = u_hat[0, :]
            u_1_hat = u_hat[1, :]
            z_0_hat = z_hat[0, :].T
            z_1_hat = z_hat[1, :].T

            # re-labelize each variable
            prod_scal_0 = np.dot(z_0_hat.flat, z_0.T.flat)
            prod_scal_1 = np.dot(z_0_hat.flat, z_1.T.flat)
            if prod_scal_0 < prod_scal_1:
                tmp = z_0_hat
                z_0_hat = z_1_hat
                z_1_hat = tmp
                tmp = u_0_hat
                u_0_hat = u_1_hat
                u_1_hat = tmp

            # error computation
            Dz_0_err = np.linalg.norm(np.diff(z_0_hat) - np.diff(z_0))
            Dz_1_err = np.linalg.norm(np.diff(z_1_hat) - np.diff(z_1))
            Dz_err = 0.5 * (Dz_0_err + Dz_1_err)
            u_0_err = np.linalg.norm(u_0_hat - u_0)
            u_1_err = np.linalg.norm(u_1_hat - u_1)
            u_err = 0.5 * (u_0_err + u_1_err)

            Dz_errs.append(Dz_err)
            u_errs.append(u_err)

        min_Dz_err = np.min(Dz_errs)
        min_u_err = np.min(u_errs)

        min_Dz_errs.append(min_Dz_err)
        min_u_errs.append(min_u_err)

    mean_min_Dz_err = np.mean(min_Dz_errs)
    std_min_Dz_err = np.std(min_Dz_errs)

    mean_min_u_err = np.mean(min_u_errs)
    std_min_u_err = np.std(min_u_errs)

    mean_min_Dz_errs.append(mean_min_Dz_err)
    std_min_Dz_errs.append(std_min_Dz_err)
    mean_min_u_errs.append(mean_min_u_err)
    std_min_u_errs.append(std_min_u_err)

    delta_t = time.strftime("%M min %S s", time.gmtime(time.time() - t0))

    print("[case SNR={:.2e}dB], mean min Dz-error {:.2f} with std {:.2f}, "
          "mean min u-error {:.2f} with std {:.2f}, c.t. : {}".format(
                                    snr, mean_min_Dz_err, std_min_Dz_err,
                                    mean_min_u_err, std_min_u_err, delta_t))

###############################################################################
# Plot activation errors
# ----------------------
snr = np.array(l_snr)

fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.set_xlabel("SNR [dB]", fontsize=18)
ax1.set_ylabel("L2 error", fontsize=18)
plt.errorbar(snr[1:], mean_min_Dz_errs[1:], yerr=std_min_Dz_errs[1:],
             color='black', linewidth=5.0, elinewidth=5.0)
plt.xticks(snr)
ax1.tick_params(labelsize=15)
plt.grid()
plt.tight_layout()
filename = 'activations_errors.pdf'
filename = os.path.join(plot_dir, filename)
print("Saving error plot under {0}".format(filename))
plt.savefig(filename, dpi=150)

###############################################################################
# Plot spatial maps errors
# ------------------------
fig, ax2 = plt.subplots(figsize=(7, 4))
ax2.set_xlabel("SNR [dB]", fontsize=18)
ax2.set_ylabel("L2 error", fontsize=18)
plt.errorbar(snr[1:], mean_min_u_errs[1:], yerr=std_min_u_errs[1:],
             color='black', linewidth=4.0, elinewidth=5.0)
plt.xticks(snr)
ax2.tick_params(labelsize=15)
plt.grid()
plt.tight_layout()
filename = 'spatial_maps_snr.pdf'
filename = os.path.join(plot_dir, filename)
print("Saving error plot under {0}".format(filename))
plt.savefig(filename, dpi=150)

###############################################################################
# Plot activation derivatives errors
# ----------------------------------
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.set_xlabel("SNR [dB]", fontsize=18)
ax1.set_ylabel("Dz l2 error", fontsize=18)
plt.errorbar(l_snr, mean_min_Dz_errs, yerr=std_min_Dz_errs, color='blue',
             linewidth=3.0, elinewidth=4.0)
plt.xticks(l_snr)
ax1.tick_params(labelsize=18)
plt.tight_layout()
filename = 'activation_derivatives_errors.pdf'
filename = os.path.join(plot_dir, filename)
print("Saving error plot under {0}".format(filename))
plt.savefig(filename, dpi=150)

###############################################################################
# Display the runtime of the script
# ---------------------------------
delta_t = time.gmtime(time.time() - t0_total)
delta_t = time.strftime("%H h %M min %S s", delta_t)
print(f"Script runs in {delta_t}")
