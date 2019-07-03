"""Atlas fetcher. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
import nibabel as nib
from nilearn import image
from nilearn.datasets import fetch_atlas_basc_multiscale_2015

def fetch_atlas_basc_12_2015():
    """ Fetch the BASC 12 rois atlas from 2015."""
    fname = fetch_atlas_basc_multiscale_2015(version='sym')['scale012']

    roi_atlas = image.load_img(fname)
    raw_data = np.ones_like(roi_atlas.get_data())
    raw_data[roi_atlas.get_data() == 0] = 0
    mask_full_brain = nib.Nifti1Image(raw_data,
                                      roi_atlas.affine,
                                      roi_atlas.header)
    return mask_full_brain, roi_atlas
