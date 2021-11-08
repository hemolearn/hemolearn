"""Functions shipped . """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: simplified BSD
# code extracted from Nilearn:
# https://github.com/nilearn/nilearn/blob/2fd66656/nilearn/image/image.py#L934
from nilearn.image import image


def binarize_img(img, threshold=0, mask_img=None):
    """Binarize an image such that its values are either 0 or 1.
    .. versionadded:: 0.8.1
    Parameters
    ----------
    img : a 3D/4D Niimg-like object
        Image which should be binarized.
    threshold : :obj:`float` or :obj:`str`
        If float, we threshold the image based on image intensities meaning
        voxels which have intensities greater than this value will be kept.
        The given value should be within the range of minimum and
        maximum intensity of the input image.
        If string, it should finish with percent sign e.g. "80%" and we
        threshold based on the score obtained using this percentile on
        the image data. The voxels which have intensities greater than
        this score will be kept. The given string should be
        within the range of "0%" to "100%".
    mask_img : Niimg-like object, default None, optional
        Mask image applied to mask the input data.
        If None, no masking will be applied.
    Returns
    -------
    :class:`~nibabel.nifti1.Nifti1Image`
        Binarized version of the given input image. Output dtype is int.
    See Also
    --------
    nilearn.image.threshold_img : To simply threshold but not binarize images.
    Examples
    --------
    Let's load an image using nilearn datasets module::
     >>> from nilearn import datasets
     >>> anatomical_image = datasets.load_mni152_template()
    Now we binarize it, generating a pseudo brainmask::
     >>> from nilearn.image import binarize_img
     >>> img = binarize_img(anatomical_image)
    """
    return image.math_img(
        "img.astype(bool).astype(int)",
        img=image.threshold_img(img, threshold, mask_img=mask_img)
    )
