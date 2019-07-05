"""Atlas fetcher. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
import numba
import nibabel as nib
from nilearn import image
from nilearn.datasets import fetch_atlas_basc_multiscale_2015


@numba.jit((numba.int64, numba.int64[:, :]), nopython=True, cache=True,
           fastmath=True)
def get_indices_from_roi(m, rois_idx):
    """ Return the indices of the ROI m of from the atlas rois_idx. """
    return np.sort(rois_idx[m, 1:rois_idx[m, 0] + 1])


def split_atlas(hrf_rois):
    """ Return a table of indices for each ROIs, a vector of labels for each
    ROIs and the number of ROIs from a dict atlas. """
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


def fetch_atlas(hrf_atlas):
    """ Fetch an anatomical atlas given its label. """
    if hrf_atlas == 'basc-007':
        mask_full_brain, atlas_rois = \
                                fetch_atlas_basc_12_2015(n_scales='scale007')
    elif hrf_atlas == 'basc-012':
        mask_full_brain, atlas_rois = \
                                fetch_atlas_basc_12_2015(n_scales='scale012')
    elif hrf_atlas == 'basc-036':
        mask_full_brain, atlas_rois = \
                                fetch_atlas_basc_12_2015(n_scales='scale036')
    elif hrf_atlas == 'basc-064':
        mask_full_brain, atlas_rois = \
                                fetch_atlas_basc_12_2015(n_scales='scale064')
    elif hrf_atlas == 'basc-122':
        mask_full_brain, atlas_rois = \
                                fetch_atlas_basc_12_2015(n_scales='scale122')
    else:
        raise ValueError("hrf_atlas should be in ['basc-007', 'basc-012', "
                         "'basc-036', 'basc-064', 'basc-122'], "
                         "got '{}'".format(hrf_atlas))
    return mask_full_brain, atlas_rois


def fetch_atlas_basc_12_2015(n_scales='scale007'):
    """ Fetch the BASC 12 rois atlas from 2015."""
    tmp = fetch_atlas_basc_multiscale_2015(version='sym')
    if n_scales == 'scale007':
        atlas_rois_fname = tmp['scale007']
    elif n_scales == 'scale012':
        atlas_rois_fname = tmp['scale012']
    elif n_scales == 'scale036':
        atlas_rois_fname = tmp['scale036']
    elif n_scales == 'scale064':
        atlas_rois_fname = tmp['scale064']
    elif n_scales == 'scale122':
        atlas_rois_fname = tmp['scale122']
    else:
        raise ValueError("n_scales should be in ['scale007', 'scale012', "
                         "'scale036', 'scale064', 'scale122'], "
                         "got '{}'".format(n_scales))
    roi_atlas = image.load_img(atlas_rois_fname)
    raw_data = np.ones_like(roi_atlas.get_data())
    raw_data[roi_atlas.get_data() == 0] = 0
    mask_full_brain = nib.Nifti1Image(raw_data,
                                      roi_atlas.affine,
                                      roi_atlas.header)
    return mask_full_brain, roi_atlas
