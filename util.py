import os
import nibabel as nib
import numpy as np
from enum import Enum

DATA_DIR = 'data'
NYU_DIR_NAME = 'NYU/'


def open_nii_img(filename):
    img = nib.load(filename).get_data()
    if len(img.shape) == 3:
        img = np.rot90(img, k=1, axes=(1, 2))
    if len(img.shape) == 4:
        img = img.transpose(3, 0, 2, 1)
        img = np.rot90(img, k=2, axes=(1, 2))
    return img


def get_img_name(subject_id, img_type):
    img_name = None
    if img_type == ImgType.STRUCTURAL_T1:
        img_name = STRUCTURAL_T1_FILE_FORMAT.format(subject=subject_id)
    elif img_type == ImgType.STRUCTURAL_GM:
        img_name = STRUCTURAL_GM_FILE_FORMAT.format(subject=subject_id)
    elif img_type == ImgType.STRUCTURAL_FILTER:
        img_name = STRUCTURAL_FILTER_FILE_FORMAT.format(subject=subject_id)
    elif img_type == ImgType.STRUCTURAL_TRANSFORM:
        img_name = STRUCTURAL_TRANSFORM_FILE_FORMAT.format(subject=subject_id)
    elif img_type == ImgType.FUNCTIONAL_BLURRED:
        img_name = FUNCTIONAL_BLURRED_FILE_FORMAT.format(subject=subject_id)
    elif img_type == ImgType.FUNCTIONAL_BANDPASS:
        img_name = FUNCTIONAL_BANDPASS_FILE_FORMAT.format(subject=subject_id)
    return img_name


class ImgType(Enum):
    STRUCTURAL_T1 = 1
    STRUCTURAL_GM = 2
    STRUCTURAL_FILTER = 3
    STRUCTURAL_TRANSFORM = 4
    FUNCTIONAL_BLURRED = 5
    FUNCTIONAL_BANDPASS = 6


STRUCTURAL_T1_FILE_FORMAT = 'wssd{subject}_session_1_anat.nii.gz'
STRUCTURAL_GM_FILE_FORMAT = 'wssd{subject}_session_1_anat_gm.nii.gz'
STRUCTURAL_FILTER_FILE_FORMAT = 'swssd{subject}_session_1_anat_gm.nii.gz'
STRUCTURAL_TRANSFORM_FILE_FORMAT = '{subject}_session_1_template.nii.gz'
FUNCTIONAL_BLURRED_FILE_FORMAT = 'snwmrda{subject}_session_1_rest_1.nii.gz'
FUNCTIONAL_BANDPASS_FILE_FORMAT = 'sfnwmrda{subject}_session_1_rest_1.nii.gz'
# https://github.com/tkipf/pygcn for ReHO data processing