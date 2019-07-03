""" Simple fMRI example: example to recover the different spontanious tasks
involved in the BOLD signal."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import shutil
import subprocess
import itertools
from joblib import Parallel, delayed, Memory
import pickle
import numpy as np
import matplotlib.pyplot as plt

from seven.learn_u_z_multi import lean_u_z_multi
from seven.data import _gen_multi_voxels
from seven.utils import check_random_state, get_unique_dirname


###############################################################################
# define data
seed = 0
tr = 1.0
snr = 1.0
len_h = n_times_atom = 30
n_atoms = 2
n_times_valid = 200
s = 0.01
p = 100
n_channels = p ** 2
tmp = _gen_multi_voxels(tr, n_times_atom, snr, s, n_times_valid, n_atoms,
                        n_channels, seed)
noisy_X, _, Lz, z, _, u, h, _, _, _, _ = tmp
noisy_X /= np.std(noisy_X)
u_0 = u[0, :]
u_1 = u[1, :]

###############################################################################
# estimation of u an z
params = dict(X=noisy_X, v=h, n_atoms=n_atoms, lbda_strategy='ratio',
              lbda=0.01, max_iter=50, get_obj=True, get_time=True,
              random_state=None, verbose=1)
nb_fit_try = 3
results = Parallel(n_jobs=-2)(delayed(lean_u_z_multi)(**params)
                              for _ in range(nb_fit_try))
l_last_pobj = np.array([res[-2][-1] for res in results])
best_run = np.argmin(l_last_pobj)
res = results[best_run]
Lz_hat, z_hat, u_hat, lbda, pobj, times = res

Lz_0_true = Lz[0, :].T
Lz_1_true = Lz[1, :].T
Lz_0_hat = Lz_hat[0, :].T
Lz_1_hat = Lz_hat[1, :].T
u_0_hat = u_hat[0, :]
u_1_hat = u_hat[1, :]

corr_00 = np.dot(Lz_0_true, Lz_0_hat)
corr_00 /= (np.dot(Lz_0_true, Lz_0_true) * np.dot(Lz_0_hat, Lz_0_hat))
corr_01 = np.dot(Lz_0_true, Lz_1_hat)
corr_01 /= (np.dot(Lz_0_true, Lz_0_true) * np.dot(Lz_1_hat, Lz_1_hat))

if corr_00 < corr_01:
    tmp = Lz_0_hat
    Lz_0_hat = Lz_1_hat
    Lz_1_hat = tmp
    tmp = u_0_hat
    u_0_hat = u_1_hat
    u_1_hat = tmp

Lz_0_hat /= np.max(np.abs(Lz_0_hat))
Lz_1_hat /= np.max(np.abs(Lz_1_hat))
Lz_0_true /= np.max(np.abs(Lz_0_true))
Lz_1_true /= np.max(np.abs(Lz_1_true))

###############################################################################
# results management
dirname = get_unique_dirname("results_toy_#")

if not os.path.exists(dirname):
    os.makedirs(dirname)

print("archiving '{0}' under '{1}'".format(__file__, dirname))
shutil.copyfile(__file__, os.path.join(dirname, __file__))


###############################################################################
# archiving results
res = dict(pobj=pobj, times=times, z_hat=z_hat, Lz_hat=Lz_hat, z=z, Lz=Lz)
filename = os.path.join(dirname, "results.pkl")
print("Pickling results under '{0}'".format(filename))
with open(filename, "wb") as pfile:
    pickle.dump(res, pfile)

###############################################################################
# plotting
# Lz
n_times = noisy_X.shape[-1]
plt.figure("Temporal atoms", figsize=(12, 5))
plt.subplot(121)
plt.plot(Lz_0_true, lw=2.0, label="True atom")
plt.plot(Lz_0_hat, lw=2.0, label="Est. atom")
x_0 = noisy_X[np.where(u_0 > 0)[0], :]
x_0 /= np.repeat(np.max(np.abs(x_0), axis=1)[:, None], n_times, 1)
t = np.arange(n_times)
mean_0 = np.mean(x_0, axis=0)
std_0 = np.std(x_0, axis=0)
borders_0 = (mean_0 - std_0, mean_0 + std_0)
plt.plot(mean_0, color='k', lw=0.5, label="Observed BOLD")
plt.fill_between(t, borders_0[0], borders_0[1], alpha=0.2, color='k')
plt.axhline(0.0, color='k', linewidth=0.5)
plt.xticks([0, n_times/2.0, n_times], fontsize=20)
plt.yticks([-1, 0, 1], fontsize=20)
plt.xlabel("Time [time-frames]", fontsize=20)
plt.legend(ncol=2, loc='lower center', fontsize=17, framealpha=0.3)
plt.title("First atom", fontsize=20)
plt.subplot(122)
plt.plot(Lz_1_true, lw=2.0, label="True atom")
plt.plot(Lz_1_hat, lw=2.0, label="Est. atom")
x_1 = noisy_X[np.where(u_1 > 0)[0], :]
x_1 /= np.repeat(np.max(np.abs(x_1), axis=1)[:, None], n_times, 1)
mean_1 = np.mean(x_1, axis=0)
std_1 = np.std(x_1, axis=0)
borders_1 = (mean_1 - std_1, mean_1 + std_1)
plt.plot(mean_1, color='k', lw=0.5, label="Observed BOLD")
plt.fill_between(t, borders_1[0], borders_1[1], alpha=0.2, color='k')
plt.axhline(0.0, color='k', linewidth=0.5)
plt.xticks([0, n_times/2.0, n_times], fontsize=20)
plt.yticks([-1, 0, 1], fontsize=20)
plt.xlabel("Time [time-frames]", fontsize=20)
plt.legend(ncol=2, loc='lower center', fontsize=17, framealpha=0.3)
plt.title("Second atom", fontsize=20)
plt.tight_layout()
filename = "Lz.pdf"
filename = os.path.join(dirname, filename)
plt.savefig(filename, dpi=150)
subprocess.call("pdfcrop {}".format(filename), shell=True)
os.rename(filename.split('.')[0]+'-crop.pdf', filename)
print("Saving plot under '{0}'".format(filename))

# U
fig, axes = plt.subplots(nrows=1, ncols=4)
l_u = [u_0.reshape(p, p), u_0_hat.reshape(p, p),
       u_1.reshape(p, p), u_1_hat.reshape(p, p)]
l_max_u = [np.max(u) for u in l_u]
max_u = np.max(l_max_u)
amax_u = np.argmax(l_max_u)
l_name = ["True map 1", "Est. map 1", "True map 2", "Est. map 2"]
l_im = []
for ax, u, name in zip(axes.flat, l_u, l_name):
    l_im.append(ax.matshow(u))
    ax.set_title(name, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.subplots_adjust(bottom=0.1, top=0.5, left=0.1, right=0.8,
                    wspace=0.3, hspace=0.2)
cbar_ax = fig.add_axes([0.83, 0.2, 0.02, 0.2])
cbar = fig.colorbar(l_im[amax_u], cax=cbar_ax)
cbar.set_ticks(np.linspace(0.0, max_u, 3))
filename = "U.pdf"
filename = os.path.join(dirname, filename)
plt.savefig(filename, dpi=150)
subprocess.call("pdfcrop {}".format(filename), shell=True)
os.rename(filename.split('.')[0]+'-crop.pdf', filename)
print("Saving plot under '{0}'".format(filename))

# pobj
t_start = 1.0e-3
min_obj = 1.0-4
plt.figure("Cost function (%)", figsize=(5, 5))
pobj -= (np.min(pobj) - min_obj)
pobj /= pobj[0]
plt.loglog(np.cumsum(times) + t_start, pobj, lw=2.0)
plt.title("Evolution of global cost-function")
plt.xlabel('time [s]')
plt.grid()
filename = "pobj.pdf"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename, dpi=150)
