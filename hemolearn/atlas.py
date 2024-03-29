"""Atlas module, to define the HRF ROIs. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import subprocess
import wget
import zipfile
import numpy as np
import numba
from nilearn import image, datasets
from hemolearn import image_nilearn


valid_scales = ['scale007', 'scale012', 'scale036', 'scale064', 'scale122',
                'scale197', 'scale325', 'scale444']


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


def fetch_harvard_vascular_atlas(sym=True, target_affine=np.diag((5, 5, 5))):
    """ Fetch the Harvard-Oxford brain atlas given its resolution.

    Parameters
    ----------
    sym : bool, (default=True), whether or not to force the atlas sto be
        symmetric
    target_affine : np.array, (default=np.diag((5, 5, 5))), affine matrix for
        the produced Nifti images

    Return
    ------
    mask : Nifti Image, full brain mask
    atlas : Nifti Image, ROIs atlas
    """
    # load Harvard-Oxford atlas
    harvard_oxford = datasets.fetch_atlas_harvard_oxford(
                                                'cort-maxprob-thr25-2mm',
                                                symmetric_split=True)
    atlas = harvard_oxford['maps']

    if sym:
        # 'resize' the atlas
        atlas = image.resample_img(atlas, target_affine,
                                   interpolation='nearest')

        # extract raw data
        atlas_raw = atlas.get_fdata()

        # gather chosen hemisphere ROIs
        label_rois = [harvard_oxford.labels.index(region_name)
                      for region_name in harvard_oxford.labels
                      if 'Right' in region_name]

        # create a mask for the chosen hemisphere
        mask_rois = [atlas_raw == label for label in label_rois]
        chosen_hemi_mask = mask_rois[0]
        for mask_roi in mask_rois[1:]:
            chosen_hemi_mask = np.add(chosen_hemi_mask, mask_roi)

        # clean intersection between hemishpere
        intersection_mask = chosen_hemi_mask & chosen_hemi_mask[::-1]
        chosen_hemi_mask[intersection_mask] = False

        # get hemispheres mask
        right_mask_img = image.new_img_like(atlas, chosen_hemi_mask)
        left_mask_img = image.new_img_like(atlas, chosen_hemi_mask[::-1])

        # re-index right hemisphere labels
        atlas_raw_right = right_mask_img.get_fdata() * atlas_raw
        labels_right = enumerate(np.unique(atlas_raw_right)[1:], start=1)
        for i, label_right in labels_right:
            # even integer for the right hemisphere
            atlas_raw_right[atlas_raw_right == label_right] = 2 * i

        # force the symmetry
        atlas_raw_left = np.copy(atlas_raw_right[::-1])

        # odd integer for the left hemisphere
        atlas_raw_left = (atlas_raw_left - 1) * left_mask_img.get_fdata()

        # cast it to a Nifti
        atlas_raw_full = atlas_raw_right + atlas_raw_left
        atlas_to_return = image.new_img_like(atlas, atlas_raw_full)

    else:
        atlas_to_return = atlas

    brain_mask = image_nilearn.binarize_img(atlas_to_return, threshold=0)

    return brain_mask, atlas_to_return


def fetch_basc_vascular_atlas(n_scales='scale007',
                              target_affine=np.diag((5, 5, 5))):
    """ Fetch the BASC brain atlas given its resolution.

    Parameters
    ----------
    hrf_atlas: str, BASC dataset name possible values are: 'scale007',
        'scale012', 'scale036', 'scale064', 'scale122', 'scale197', 'scale325',
        'scale444'

    target_affine : np.array, (default=np.diag((5, 5, 5))), affine matrix for
        the produced Nifti images

    Return
    ------
    mask_full_brain : Nifti Image, full mask brain
    atlas_rois : Nifti Image, ROIs atlas
    """
    if n_scales not in valid_scales:
        raise ValueError(f"n_scales should be in {valid_scales}, "
                         f"got '{n_scales}'")

    basc_dataset = datasets.fetch_atlas_basc_multiscale_2015(version='sym')
    atlas_rois_fname = basc_dataset[n_scales]
    atlas_to_return = image.load_img(atlas_rois_fname)

    atlas_to_return = image.resample_img(atlas_to_return, target_affine,
                                         interpolation='nearest')

    brain_mask = image_nilearn.binarize_img(atlas_to_return, threshold=0)

    return brain_mask, atlas_to_return


def fetch_aal_vascular_atlas(target_affine=np.diag((5, 5, 5))):
    """ Fetch the AAL brain atlas given its resolution.

    Parameters
    ----------
    target_affine : np.array, (default=np.diag((5, 5, 5))), affine matrix for
        the produced Nifti images

    Return
    ------
    mask_full_brain : Nifti Image, full mask brain
    atlas_rois : Nifti Image, ROIs atlas
    """
    aal_dataset = datasets.fetch_atlas_aal()
    atlas_rois_fname = aal_dataset.maps
    atlas_to_return = image.load_img(atlas_rois_fname)

    atlas_to_return = image.resample_img(atlas_to_return, target_affine,
                                         interpolation='nearest')

    brain_mask = image_nilearn.binarize_img(atlas_to_return, threshold=0)

    return brain_mask, atlas_to_return


def fetch_aal3_vascular_atlas(target_affine=np.diag((5, 5, 5))):
    """ Fetch the AAL3 brain atlas given its resolution.

    Parameters
    ----------
    target_affine : np.array, (default=np.diag((5, 5, 5))), affine matrix for
        the produced Nifti images

    Return
    ------
    mask_full_brain : Nifti Image, full mask brain
    atlas_rois : Nifti Image, ROIs atlas
    """
    data_dir = os.path.join(os.path.expanduser('~'), 'hemolearn_data')
    aal3_dir = os.path.join(data_dir, 'AAL3v1')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(aal3_dir):
        os.makedirs(aal3_dir)

    url = 'https://www.gin.cnrs.fr/wp-content/uploads/AAL3v1_for_SPM12.zip'
    dest_filename = os.path.join(aal3_dir, wget.filename_from_url(url))

    if not os.path.exists(os.path.join(aal3_dir, 'AAL3')):
        # download files
        wget.download(url, out=aal3_dir)

        # extract files
        with zipfile.ZipFile(dest_filename, 'r') as zip_ref:
            zip_ref.extractall(aal3_dir)

        # clean directory
        cmd = (f"find {data_dir} -type f \( -iname \*.m -o "  # noqa: W605
               f"-iname \*.zip -o -iname \*.rtf -o -iname "  # noqa: W605
               f"\*.pdf \) -delete")  # noqa: W605
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)

    atlas_fname = os.path.join(aal3_dir, 'AAL3', 'AAL3v1.nii.gz')
    atlas_to_return = image.load_img(atlas_fname)

    atlas_to_return = image.resample_img(atlas_to_return, target_affine,
                                         interpolation='nearest')

    brain_mask = image_nilearn.binarize_img(atlas_to_return, threshold=0)

    return brain_mask, atlas_to_return
