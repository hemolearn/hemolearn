""" Plotting utilities. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import time
import subprocess
import collections
import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting, image

from .atlas import fetch_vascular_atlas, fetch_atlas_basc_2015, split_atlas
from .utils import tp, fwhm


def plotting_obj_values(times, pobj, plot_dir='.', fname=None, min_obj=1.0-6,
                        figsize=(3, 3), verbose=False):
    """ Plot, and save as pdf, the objective function values.

    Parameters
    ----------
    times : array, shape (current_n_iter,) or(3 * current_n_iter,), the saved
        cost-function
    pobj : array, shape (current_n_iter,) or(3 * current_n_iter,), the saved
        duration per steps
    plot_dir : str, (default='.'), directory under which the pdf is saved
    fname : str, (default='obj.pdf'), filename under which the pdf is saved
    min_obj : float, (default=1.0-6), tolerance for minimum of the
        cost-function
    figsize : tuple of int, (default=(3, 3)), size of the produced figure
    verbose : bool, (default=False), verbosity level
    """
    plt.figure("Cost function (%)", figsize=figsize, constrained_layout=True)
    pobj -= (np.min(pobj) - min_obj)
    pobj /= pobj[0]
    plt.plot(np.cumsum(times), pobj, lw=3.0)
    plt.title("Evolution of\nglobal cost-function")
    plt.xlabel('Time [s]')
    plt.ylabel('cost function [%]')
    plt.grid()
    if fname is None:
        fname = 'obj.pdf'
    fname = os.path.join(plot_dir, fname)
    if verbose:
        print("Saving plot under '{0}'".format(fname))
    plt.savefig(fname, dpi=150)


def plotting_temporal_comp(z, variances, t_r, onset=False, plot_dir='.',
                           fname=None, aux_plots=None, aux_plots_kwargs=dict(),
                           verbose=False):
    """ Plot, and save as pdf, each temporal estimated component.

    Parameters
    ----------
    z : array, shape (n_atoms, n_times_valid), the temporal components
    variances : array, shape (n_atoms, ) the order variances for each
        components
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    onset : bool, (default=False), whether or not to plot the first order
        derivative of z
    plot_dir : str, (default='.'), directory under which the pdf is saved
    fname : str, (default='z.pdf'), filename under which the pdf is saved
    aux_plots : func, to plot a additional features on the figure
    aux_plots_kwargs : dict, keywords arguments for the aux_plots func
    verbose : bool, (default=False), verbosity level
    """
    n_atoms, n_times_valid = z.shape
    if onset:
        plt.figure("Onset Temporal atoms", figsize=(8, 5 * n_atoms))
    else:
        plt.figure("Temporal atoms", figsize=(8, 5 * n_atoms))
    _xticks = [0, int(n_times_valid / 2.0), int(n_times_valid)]
    _xticks_labels = [
        0,
        time.strftime("%Mm%Ss", time.gmtime(int(t_r * n_times_valid / 2.0))),
        time.strftime("%Mm%Ss", time.gmtime(int(t_r * n_times_valid)))
        ]
    for k in range(1, n_atoms + 1):
        expl_var = variances[k - 1]
        z_k = z[k - 1, :].T
        plt.subplot(n_atoms, 1, k)
        if onset:
            plt.stem(np.diff(z_k))
            if fname is None:
                fname = "Dz.pdf"
        else:
            plt.plot(z_k, lw=2.0)
            if fname is None:
                fname = "z.pdf"
        plt.axhline(0.0, color='black', lw=2.0)
        if aux_plots is not None:
            aux_plots(**aux_plots_kwargs)
        plt.title("Atom-{} (explained variance = {:.2e})".format(k, expl_var),
                  fontsize=20)
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Signal change [%]", fontsize=20)
        plt.xticks(_xticks, _xticks_labels, fontsize=20)
        plt.yticks([np.min(z_k), 0, np.max(z_k)], fontsize=20)
        plt.grid()
    plt.tight_layout()
    filename = os.path.join(plot_dir, fname)
    if verbose:
        print("Saving plot under '{0}'".format(filename))
    plt.savefig(filename, dpi=150)


def plotting_spatial_comp(u, variances, masker, plot_dir='.', fname=None,
                          display_mode='ortho', perc_voxels_to_retain=0.1,
                          bg_img=None, save_nifti=False, verbose=False):
    """ Plot, and save as pdf, each spatial estimated component.

    Parameters
    ----------
    u : array, shape (n_atoms, n_voxels), the spatial maps
    variances : array, shape (n_atoms, ) the order variances for each
        components
    masker : Nilearn-Masker like, masker class to perform the inverse Nifti
        transformation
    plot_dir : str, (default='.'), directory under which the pdf is saved
    fname : str, (default='u.pdf'), filename under which the pdf is saved
    display_mode : None or str, coords to cut the plotting, possible value are
        None to have x, y, z or 'x', 'y', 'z' for a single cut
    perc_voxels_to_retain : float, (default=0.1), percentage of voxels to
        retain when plotting the spatial maps
    bg_img : Nifti-like or None, (default=None), background image, None means
        no image
    save_nifti : bool, (default=False), whether or not to save the image as
        Nifti
    verbose : bool, (default=False), verbosity level
    """
    if display_mode in ['x', 'y', 'z']:
        cut_coords = 1
        colorbar = False
        compress_plot = True
    else:
        display_mode = 'ortho'
        cut_coords = None
        colorbar = True
        compress_plot = False

    n_atoms, n_voxels = u.shape
    img_u = []
    for k in range(1, n_atoms + 1):
        u_k = u[k - 1]
        last_retained_voxel_idx = int(perc_voxels_to_retain * n_voxels)
        th = np.sort(u_k)[-last_retained_voxel_idx]
        expl_var = variances[k - 1]
        if compress_plot:
            title = "Map-{}".format(k)
        else:
            title = "Map-{} (explained variance = {:.2e})".format(k, expl_var)
        img_u_k = masker.inverse_transform(u_k)
        img_u.append(img_u_k)
        if bg_img is not None:
            plotting.plot_stat_map(img_u_k, title=title, colorbar=colorbar,
                                   display_mode=display_mode,
                                   cut_coords=cut_coords, threshold=th,
                                   bg_img=bg_img)
        else:
            plotting.plot_stat_map(img_u_k, title=title, colorbar=colorbar,
                                   display_mode=display_mode,
                                   cut_coords=cut_coords, threshold=th)
        if save_nifti:
            nii_filename = os.path.join(plot_dir, "u_{0:03d}.nii".format(k))
            img_u_k.to_filename(nii_filename)
        plt.savefig(os.path.join(plot_dir, "u_{0:03d}.pdf".format(k)), dpi=150)
    pdf_files = os.path.join(plot_dir, 'u_*.pdf')
    if fname is None:
        fname = 'u.pdf'
    pdf_file = os.path.join(plot_dir, fname)
    if compress_plot:
        cmd_cat = ("pdfjam --suffix nup --nup 8x5 --no-landscape {} "
                   "--outfile {}".format(pdf_files, pdf_file))
        subprocess.call(cmd_cat, shell=True)
        cmd_crop = "pdfcrop {0} {0}".format(pdf_file)
        subprocess.call(cmd_crop, shell=True)
    else:
        cmd = "pdftk {} cat output {}".format(pdf_files, pdf_file)
        subprocess.call(cmd, shell=True)
    subprocess.call("rm -f {}".format(pdf_files), shell=True)
    if verbose:
        print("Saving plot under '{0}'".format(pdf_file))


def plotting_hrf(v, t_r, hrf_ref=None, masker=None, atlas_type=None,
                 atlas_kwargs=dict(), n_scales=122, normalized=False,
                 plot_dir='.', fname=None, verbose=True):
    """ Plot, and save as pdf, each HRF for each ROIs.

    Parameters
    ----------
    v : array, shape (n_hrf_rois, n_times_atom), the initial used HRFs
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    hrf_ref : array or None, shape (n_times_atom, ), (default=None), reference
        HRF to plot for comparison
    masker : Nilearn-Masker like, masker class to perform the inverse Nifti
        transformation
    atlas_type : str, func, or None, (default=None), atlas type, possible
        choice are ['havard', 'basc', given-function]
    atlas_kwargs : dict, (default=dict()), additional kwargs for the atlas, if
        a function is passed
    n_scales : str, (default='scale122'), select the number of scale if
        hrf_atlas == 'basc'.
    normalized : bool, (default=False), whether or not to normalized by the
        l-inf norm each HRFs
    plot_dir : str, (default='.'), directory under which the pdf is saved
    fname : str, (default='v.pdf'), filename under which the pdf is saved
    verbose : bool, (default=False), verbosity level
    """
    if atlas_type == 'havard':
        _, atlas_rois = fetch_vascular_atlas()
    elif atlas_type == 'basc':
        n_scales_ = f"scale{int(n_scales)}"
        _, atlas_rois = fetch_atlas_basc_2015(n_scales=n_scales_)
    elif isinstance(atlas_type, collections.Callable):
        _, atlas_rois = atlas_type(**atlas_kwargs)
    else:
        raise ValueError(f"atlas_type should belong to ['havard', 'basc', "
                         f"given-function], got {atlas_type}")
    hrf_rois = dict()
    rois = masker.transform(atlas_rois).astype(int).ravel()
    index = np.arange(rois.shape[-1])
    for roi_label in np.unique(rois):
        hrf_rois[roi_label] = index[roi_label == rois]
    _, roi_label_from_hrf_idx, _ = split_atlas(hrf_rois)
    n_rois, n_times_atoms = v.shape
    n_cols = np.int(np.sqrt(n_rois))
    n_raws = n_rois // n_cols
    n_cols += 1 if (n_rois % n_cols != 0) else 0
    plotting.plot_roi(atlas_rois, title="ROIs", cmap=plt.cm.gist_ncar,
                      figure=plt.figure(figsize=(n_cols * 2, n_raws)))
    plt.savefig(os.path.join(plot_dir, "rois_.pdf"), dpi=150)
    fig, axis = plt.subplots(n_raws, n_cols, sharex=True, sharey=True,
                             figsize=(n_cols * 2, n_raws * 2))
    axis = axis.ravel()
    _xticks = [0, int(n_times_atoms / 2.0), int(n_times_atoms)]
    _xticks_labels = [0, np.round(t_r * int(n_times_atoms / 2.0), 2),
                      np.round(t_r * int(n_times_atoms), 2)]
    x_text, y_text = 0.0, np.min(v)
    for i in range(n_rois):
        color_ = plt.cm.gist_ncar(roi_label_from_hrf_idx[i] / n_rois)
        v_ = v[i, :]
        if normalized:
            v_ /= np.max(np.abs(v[i, :]))
        axis[i].plot(v_, lw=3.0, c=color_)
        text_ = "TP={0:.2f}s\nFWHM={1:.2f}s".format(tp(t_r, v[i, :]),
                                                    fwhm(t_r, v[i, :]))
        axis[i].text(x_text, y_text, text_)
        if normalized:
            hrf_ref /= np.max(np.abs(hrf_ref))
        if hrf_ref is not None:
            axis[i].plot(hrf_ref, '--k', lw=0.5)
        axis[i].grid(True)
    plt.xticks(_xticks, _xticks_labels)
    plt.yticks([np.min(v), 0, np.max(v)])
    fig.text(0.5, 0.04, 'Time [s]', ha='center', fontsize=2 * (n_cols * 2))
    fig.text(0.04, 0.5, 'Signal change [-]', va='center', rotation='vertical',
             fontsize=2 * (n_cols * 2))
    fig.suptitle("HRFs", fontsize=5 * (n_cols * 2))
    plt.savefig(os.path.join(plot_dir, "hrf_.pdf"), dpi=150)
    pdf_files = os.path.join(plot_dir, 'rois_.pdf')
    pdf_files += ' '
    pdf_files += os.path.join(plot_dir, 'hrf_.pdf')
    if fname is None:
        fname = 'v.pdf'
    pdf_file = os.path.join(plot_dir, fname)
    subprocess.call("pdftk {} cat output {}".format(pdf_files, pdf_file),
                    shell=True)
    subprocess.call("rm -f {}".format(pdf_files), shell=True)
    if verbose:
        print("Saving plot under '{0}'".format(pdf_file))


def plotting_hrf_stats(v, t_r, hrf_ref=None, stat_type='tp',
                       display_mode='ortho', cut_coords=None,
                       masker=None, atlas_type='havard', atlas_kwargs=dict(),
                       n_scales=122, plot_dir='.', fname=None,
                       save_nifti=False, verbose=False):
    """ Plot, and save as pdf, each stats HRF for each ROIs.

    Parameters
    ----------
    v : array, shape (n_hrf_rois, n_times_atom), the initial used HRFs
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    hrf_ref : array or None, shape (n_times_atom, ), (default=None), reference
        HRF to plot for comparison
    stat_type : str, (default='tp'), statistic to compute on each HRFs possible
        choice are ('tp', 'fwhm')
    normalized : bool, (default=False), whether or not to normalized by the
        l-inf norm each HRFs
    display_mode : None or str, coords to cut the plotting, possible value are
        None to have x, y, z or 'x', 'y', 'z' for a single cut
    cut_coords : tuple or None, MNI coordinate to perform display
    masker : Nilearn-Masker like, masker class to perform the inverse Nifti
        transformation
    atlas_type : str, func, or None, (default=None), atlas type, possible
        choice are ['havard', 'basc', given-function]
    atlas_kwargs : dict, (default=dict()), additional kwargs for the atlas,
        if a function is passed.
    n_scales : int, (default=122), number of scale if atlas_type == 'basc'
    plot_dir : str, (default='.'), directory under which the pdf is saved
    fname : str, (default='v_{fwhm/tp}.pdf'), filename under which the pdf is
        saved
    save_nifti : bool, (default=False), whether or not to save the image as
        Nifti
    verbose : bool, (default=False), verbosity level
    """
    if stat_type not in ['tp', 'fwhm']:
        raise ValueError("stat_type should be in ['tp', 'fwhm'], "
                         "got {}".format(stat_type))
    if atlas_type == 'havard':
        _, atlas_rois = fetch_vascular_atlas()
    elif atlas_type == 'basc':
        n_scales_ = f"scale{int(n_scales)}"
        _, atlas_rois = fetch_atlas_basc_2015(n_scales=n_scales_)
    elif isinstance(atlas_type, collections.Callable):
        _, atlas_rois = atlas_type(**atlas_kwargs)
    else:
        raise ValueError(f"atlas_type should belong to ['havard', 'basc', "
                         f"given-function], got {atlas_type}")
    hrf_rois = dict()
    rois = masker.transform(atlas_rois).astype(int).ravel()
    index = np.arange(rois.shape[-1])
    for roi_label in np.unique(rois):
        hrf_rois[roi_label] = index[roi_label == rois]
    _, roi_label_from_hrf_idx, _ = split_atlas(hrf_rois)
    raw_atlas_rois = atlas_rois.get_data()
    n_hrf_rois, n_times_atom = v.shape
    if hrf_ref is not None:
        if stat_type == 'tp':
            ref_stat = tp(t_r, hrf_ref)
        elif stat_type == 'fwhm':
            ref_stat = fwhm(t_r, hrf_ref)
    for m in range(n_hrf_rois):
        v_ = v[m, :]
        if stat_type == 'tp':
            stat_ = tp(t_r, v_)
            stat_name = 'TtP'
        elif stat_type == 'fwhm':
            stat_ = fwhm(t_r, v_)
            stat_name = 'FWHM'
        if hrf_ref is not None:
            stat_ -= ref_stat
            title = "{0}(-{0}-reference) map (s)".format(stat_name)
        else:
            title = "{} map (s)".format(stat_name)
        label = roi_label_from_hrf_idx[m]
        raw_atlas_rois[raw_atlas_rois == label] = stat_
    stats_map = image.new_img_like(atlas_rois, raw_atlas_rois)
    plotting.plot_stat_map(stats_map, title=title, colorbar=True,
                           display_mode=display_mode, cut_coords=cut_coords,
                           symmetric_cbar=False)
    if save_nifti:
        nii_filename = os.path.join(plot_dir, "v_{}.nii".format(stat_type))
        stats_map.to_filename(nii_filename)
    if fname is None:
        fname = "v_{}.pdf".format(stat_type)
    fname = os.path.join(plot_dir, fname)
    plt.savefig(fname, dpi=150)
    if verbose:
        print("Saving plot under '{0}'".format(fname))
