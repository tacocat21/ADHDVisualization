from torchvision import transforms
import torch
import numpy as np
import pandas as pd
import util
import torch.utils.data
import os

import ipdb

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, img_type, transform=None):
        super(ImageDataset, self).__init__()
        self.base_dir = os.path.join(util.DATA_DIR, base_dir)
        phenotype_filename = os.path.join(util.DATA_DIR, base_dir, base_dir.replace('/', '') + '_phenotypic.csv')
        self.phenotype_info = pd.read_csv(phenotype_filename)
        self.img_type = img_type
        self.transform = transform

    def change_img_type(self, img_type):
        self.img_type = img_type

    def __getitem__(self, idx):
        info = self.phenotype_info.iloc[idx]
        dir_name = os.path.join(self.base_dir, str(info[0]))
        try:
            img_name = util.get_img_name(subject_id=info[0], img_type=self.img_type)
            img = util.open_nii_img(os.path.join(dir_name, img_name))
        except:
            return self.__getitem__((idx +1) % self.__len__())
        if len(img.shape) == 3:
            img = img[:, 20:, 25:220]
            img = util.resize_3d_img(img, (util.MODEL_IMG_INPUT_SIZE, util.MODEL_IMG_INPUT_SIZE))
            if self.transform:
                res = []
                for i in range(img.shape[0]):
                    res.append(self.transform(img[i]))
                img = torch.stack(res, axis=0)
        elif len(img.shape) == 4:
            raise NotImplementedError("Need to imploment 4d access")
        return img, info['DX']

    def __len__(self):
        return len(self.phenotype_info)


train_transform = transforms.Compose([transforms.ToTensor()])
# , transforms.Normalize((0), (1))
def get_data_loader(base_dir, img_type, transform=None, shuffle=True):
    dataset = ImageDataset(base_dir, img_type, transform=transform)
    return torch.utils.data.dataloader.DataLoader(dataset, num_workers=32, shuffle=shuffle)
