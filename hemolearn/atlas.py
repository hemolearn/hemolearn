"""Atlas module, to define the HRF ROIs. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
import numba
import nibabel as nib
from nilearn import image
from nilearn.datasets import fetch_atlas_basc_multiscale_2015


@numba.jit((numba.int64, numba.int64[:, :]), nopython=True, cache=True,
           fastmath=True)
def get_indices_from_roi(m, rois_idx):  # pragma: no cover
    """ Return the indices of the ROI of index m from the given atlas rois_idx.

    Parameters
    ----------
    m : int, index of the ROI
    rois_idx : int array, shape (n_hrf_rois, max_n_voxels_roi) voxels indices
        for each ROI

    Return
    ------
    roi_indices : int array, shape (n_voxels_roi,) indices of voxels of the ROI
    """
    return np.sort(rois_idx[m, 1:rois_idx[m, 0] + 1])


def split_atlas(hrf_rois):
    """ Split the HRF atlas into a table of indices for each ROIs, a vector of
    labels for each ROIs and the number of ROIs from a dict atlas.

    Parameters
    ----------
    hrf_rois : dict (key: ROIs labels, value: indices of voxels of the ROI)
        atlas HRF

    Return
    ------
    rois_idx : int array, shape (n_hrf_rois, max_n_voxels_roi) voxels indices
        for each ROI
    rois_label : int array, shape (n_hrf_rois,) label for each ROI
    n_hrf_rois : int, number of ROIs in the HRF atlas
    """
    # main use of the function is to be passed to Numba (C-object compliant)
    n_hrf_rois = len(hrf_rois)
    len_indices = [len(indices) for indices in hrf_rois.values()]
    max_voxels_in_rois = np.max(len_indices)
    rois_idx = np.empty((n_hrf_rois, max_voxels_in_rois + 1), dtype=int)
    rois_label = np.empty((n_hrf_rois,))
    for i, items in enumerate(hrf_rois.items()):
        label, indices = items
        rois_label[i] = label
        rois_idx[i, 0] = len(indices)
        rois_idx[i, 1: len(indices) + 1] = indices
    return rois_idx, rois_label, n_hrf_rois


def fetch_atlas_basc_2015(n_scales='scale007'):
    """ Fetch the BASC brain atlas given its resolution.

    Parameters
    ----------
    hrf_atlas: str, BASC dataset name possible values are: 'scale007',
        'scale012', 'scale036', 'scale064', 'scale122'

    Return
    ------
    mask_full_brain : Nifti Image, full mask brain
    atlas_rois : Nifti Image, ROIs atlas
    """
    basc_dataset = fetch_atlas_basc_multiscale_2015(version='sym')
    if n_scales == 'scale007':
        atlas_rois_fname = basc_dataset['scale007']
    elif n_scales == 'scale012':
        atlas_rois_fname = basc_dataset['scale012']
    elif n_scales == 'scale036':
        atlas_rois_fname = basc_dataset['scale036']
    elif n_scales == 'scale064':
        atlas_rois_fname = basc_dataset['scale064']
    elif n_scales == 'scale122':
        atlas_rois_fname = basc_dataset['scale122']
    else:
        raise ValueError("n_scales should be in ['scale007', 'scale012', "
                         "'scale036', 'scale064', 'scale122'], "
                         "got '{}'".format(n_scales))
    roi_atlas = image.load_img(atlas_rois_fname)
    # compute the full brain mask (0/1 Nifti object)
    raw_data = np.ones_like(roi_atlas.get_data())
    raw_data[roi_atlas.get_data() == 0] = 0
    mask_full_brain = nib.Nifti1Image(raw_data,
                                      roi_atlas.affine,
                                      roi_atlas.header)
    return mask_full_brain, roi_atlas
