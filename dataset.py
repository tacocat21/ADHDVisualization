import torch
import numpy as np
import pandas as pd
import util
import torch.utils.data
import os

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, img_type, transform=None):
        super(ImageDataset, self).__init__()
        self.base_dir = base_dir
        phenotype_filename = os.path.join(util.DATA_DIR, base_dir, base_dir.replace('/', '') + '_phenotypic.csv')
        self.phenotype_info = pd.read_csv(phenotype_filename)
        self.img_type = img_type
        self.transform = transform

    def change_img_type(self, img_type):
        self.img_type = img_type

    def __getitem__(self, idx):
        info = self.phenotype_info.iloc[idx]
        dir_name = os.path.join(self.base_dir, info[0])
        img_name = util.get_img_name(subject_id=info[0], img_type=self.img_type)
        img = util.open_nii_img(os.path.join(dir_name, img_name))
        if self.transform is not None:
            img = self.transform(img)
        return img, info['DX']

    def __len__(self):
        return len(self.phenotype_info)


def get_data_loader(base_dir, img_type, transform=None, shuffle=True):
    dataset = ImageDataset(base_dir, img_type, transform=transform)
    return torch.utils.data.dataloader.DataLoader(dataset, num_workers=32, shuffle=shuffle)
