import pickle
import os
import nibabel as nib
import numpy as np
from enum import Enum
import scipy.misc
import ipdb
import glob
import model
import torch
IMG_LENGTH = 187
BATCH_SIZE = 25
DATA_DIR = 'data/images'
NYU_DIR_NAME = 'NYU/'
MODEL_IMG_INPUT_SIZE=(169, 195)
# 'model/ImgType.STRUCTURAL_FILTER/adam/0.0001/29.ckpt'
def load_model(filename):
    _model = model.StructuralModel3DFullImage()
    old_model = torch.load(filename)
    _model.load_state_dict(old_model.state_dict())
    _model = _model.cuda()
    return _model

def mkdir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        pass


def open_nii_img(filename):
    img = nib.load(filename).get_data()
    if len(img.shape) == 3:
        img = np.rot90(img, k=1, axes=(1, 2))
    if len(img.shape) == 4:
        img = img.transpose(3, 0, 2, 1)
        img = np.rot90(img, k=2, axes=(1, 2))
    return img

def resize_3d_img(img, shape, normalize=True):
    res = []
    img = img.astype(np.float32)
    # ipdb.set_trace()
    for i in range(img.shape[0]):
        new_img = scipy.misc.imresize(img[i], shape)
        # normalize to 0 mean, 1 std
        if normalize:
            new_img = (new_img - np.mean(new_img))
            if np.std(img[i]) > 0.00001:
                new_img = new_img/np.std(img[i])
        # else:
        #     continue
        res.append(new_img)
    return np.asarray(res)

def format_subject_id_name(subject_id):
    subject_id = str(subject_id)
    if '.' in subject_id:
        subject_id = subject_id[:subject_id.find('.')]
    if len(subject_id) < 7:
        subject_id = '0' * (7 - len(subject_id)) + subject_id
    return subject_id

def get_img_name_list(subject_id, img_type, dir_name):
    img_name = None
    if img_type == ImgType.STRUCTURAL_T1:
        img_name = STRUCTURAL_T1_FILE_FORMAT.format(subject=subject_id, session_id='*')
    elif img_type == ImgType.STRUCTURAL_GM:
        img_name = STRUCTURAL_GM_FILE_FORMAT.format(subject=subject_id, session_id='*')
    elif img_type == ImgType.STRUCTURAL_FILTER:
        img_name = STRUCTURAL_FILTER_FILE_FORMAT.format(subject=subject_id, session_id='*')
    elif img_type == ImgType.STRUCTURAL_TRANSFORM:
        img_name = STRUCTURAL_TRANSFORM_FILE_FORMAT.format(subject=subject_id, session_id='*')
    elif img_type == ImgType.FUNCTIONAL_BLURRED:
        img_name = FUNCTIONAL_BLURRED_FILE_FORMAT.format(subject=subject_id, session_id='*')
    elif img_type == ImgType.FUNCTIONAL_BANDPASS:
        img_name = FUNCTIONAL_BANDPASS_FILE_FORMAT.format(subject=subject_id, session_id='*')
    path = os.path.join(dir_name, img_name)
    try:
        return glob.glob(path)
    except IndexError:
        print("Error obtaining in {}".format(path))
        return []


class ImgType(Enum):
    STRUCTURAL_T1 = 1
    STRUCTURAL_GM = 2
    STRUCTURAL_FILTER = 3
    STRUCTURAL_TRANSFORM = 4
    FUNCTIONAL_BLURRED = 5
    FUNCTIONAL_BANDPASS = 6


STRUCTURAL_T1_FILE_FORMAT = 'wssd{subject}_session_{session_id}_anat.nii.gz'
STRUCTURAL_GM_FILE_FORMAT = 'wssd{subject}_session_{session_id}_anat_gm.nii.gz'
STRUCTURAL_FILTER_FILE_FORMAT = 'swssd{subject}_session_{session_id}_anat_gm.nii.gz'
STRUCTURAL_TRANSFORM_FILE_FORMAT = '{subject}_session_{session_id}_template.nii.gz'
FUNCTIONAL_BLURRED_FILE_FORMAT = 'snwmrda{subject}_session_{session_id}_rest_{rest_id}.nii.gz'
FUNCTIONAL_BANDPASS_FILE_FORMAT = 'sfnwmrda{subject}_session_{session_id}_rest_{rest_id}.nii.gz'
# https://github.com/tkipf/pygcn for ReHO data processing

def save_data(d, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj=d, file=fp)

