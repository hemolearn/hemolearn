""" Plotting utilities. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import time
import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting, datasets

from .utils import th


def plot_temporal_activations(z, t_r, onset=False, filename='activations.png',
                              aux_plots=None, aux_plots_kwargs=dict(),
                              verbose=False):
    """ Plot temporal activations.

    Parameters
    ----------
    z : array, shape (n_atoms, n_times_valid), the temporal components
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    onset : bool, (default=False), whether or not to plot the first order
        derivative of z
    plot_dir : str, (default='.'), directory under which the pdf is saved
    filename : str, (default='z.pdf'), filename under which the pdf is saved
    aux_plots : func, to plot a additional features on the figure
    aux_plots_kwargs : dict, keywords arguments for the aux_plots func
    verbose : bool, (default=False), verbosity level
    """
    n_atoms, n_times_valid = z.shape

    ncols = 3
    nrows = int(np.ceil(n_atoms / ncols) + 1)
    _, axis = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 3))

    _xticks = [0, int(n_times_valid / 2.0), int(n_times_valid)]
    _xticks_labels = [
        0,
        time.strftime("%Mm%Ss", time.gmtime(int(t_r * n_times_valid / 2.0))),
        time.strftime("%Mm%Ss", time.gmtime(int(t_r * n_times_valid)))
        ]

    k = 1
    for i in range(nrows):
        for j in range(ncols):

            if k > n_atoms:
                axis[i, j].set_axis_off()

            else:
                z_k = z[k - 1, :].T

                if onset:
                    axis[i, j].stem(np.diff(z_k))
                else:
                    axis[i, j].plot(z_k, lw=2.0)
                axis[i, j].axhline(0.0, color='black', lw=2.0)

                if aux_plots is not None:
                    aux_plots(**aux_plots_kwargs)

                axis[i, j].set_title(f"Activation-{k}", fontsize=20)
                axis[i, j].set_xlabel("Time", fontsize=18)
                axis[i, j].set_xticks(_xticks)
                axis[i, j].set_xticklabels(_xticks_labels, fontsize=18)
                axis[i, j].set_yticks([np.min(z_k), 0, np.max(z_k)])
                axis[i, j].set_yticklabels([f"{np.min(z_k):.1e}",
                                            f"{0:.1f}",
                                            f"{np.max(z_k):.1e}"], fontsize=18)

            k += 1

    if verbose:
        print(f"Saving plot under '{filename}'")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)


def plot_spatial_maps(u_img, filename='spatial_maps.png',
                      display_mode='ortho', cut_coords=None,
                      perc_voxels_to_retain='10%',
                      bg_img=datasets.MNI152_FILE_PATH, verbose=False):
    """ Plot spatial maps.

    Parameters
    ----------
    u_img : list of Nifti, the spatial maps
    filename : str, (default='u.pdf'), filename under which the pdf is saved
    display_mode : None or str, coords to cut the plotting, possible value are
        None to have x, y, z or 'x', 'y', 'z' for a single cut
    cut_coords : list, list of slice coordinates to display
    perc_voxels_to_retain : float, (default=0.1), percentage of voxels to
        retain when plotting the spatial maps
    bg_img : Nifti-like or None, (default=None), background image, None means
        no image
    verbose : bool, (default=False), verbosity level
    """
    n_atoms = len(u_img)

    ncols = 3
    nrows = int(np.ceil(n_atoms / ncols) + 1)
    _, axis = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 3))

    k = 1
    for i in range(nrows):
        for j in range(ncols):

            if k > n_atoms:
                axis[i, j].set_axis_off()

            else:
                t = th(u_img[k - 1].get_fdata(), t=perc_voxels_to_retain,
                       absolute=False)
                plotting.plot_stat_map(u_img[k - 1], title=f"Map-{k}",
                                       colorbar=True, axes=axis[i, j],
                                       display_mode=display_mode,
                                       cut_coords=cut_coords, threshold=t,
                                       bg_img=bg_img)

                k += 1

    if verbose:
        print(f"Saving plot under '{filename}'")

    plt.savefig(filename, dpi=150)


def plot_vascular_map(a_img, display_mode='ortho', cut_coords=None,
                      filename='vascular_map.png', vmax=None, verbose=False):
    """ Plot vascular map.

    Parameters
    ----------
    a_img : list of Nifti, the vascular maps
    display_mode : None or str, coords to cut the plotting, possible value are
        None to have x, y, z or 'x', 'y', 'z' for a single cut
    cut_coords : tuple or None, MNI coordinate to perform display
    filename : str, (default='vascular_map.png'), filename under which the pdf
        is saved
    vmax : float, (default=None), maximum of the colorbar.
    verbose : bool, (default=False), verbosity level
    """
    if vmax is None:
        vmax = a_img.get_fdata().max()
    plotting.plot_stat_map(a_img, title="Vascular map", colorbar=True,
                           vmax=vmax, display_mode=display_mode,
                           cut_coords=cut_coords)
    plt.savefig(filename, dpi=150)

    if verbose:
        print(f"Saving plot under '{filename}'")
