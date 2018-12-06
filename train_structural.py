import ipdb
import numpy as np
import os
import sys
import time
import util
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model
#from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist
import torchvision
import dataset
import random
import resnet_3d

import h5py
# import cv2

from multiprocessing import Pool


IMAGE_SIZE = 224
NUM_CLASSES = 4
lr = 0.005
num_of_epochs = 30


def run(optimizer_type, img_type):
    # _model = resnet_3d.resnet18(sample_size=IMAGE_SIZE, sample_duration=util.IMG_LENGTH)
    # _model.fc = torch.nn.Linear(_model.fc.in_features, NUM_CLASSES)
    _model = model.StructuralModel3D()
    _model = _model.cuda()
    if optimizer_type == 'adam':
        optimizer = optim.Adam(_model.parameters(), lr=lr)
    # ipdb.set_trace()
    save_dir = 'model/{}/{}/{}'.format(str(img_type), optimizer_type, lr)
    util.mkdir(save_dir)
    # base_dirs = ['Peking_1', 'NYU', 'Peking_2', 'OHSU', 'WashU', 'KKI', 'NeuroIMAGE']
    base_dirs = ['NYU', 'WashU', 'NeuroIMAGE']
    criterion = nn.CrossEntropyLoss()
    count = 0
    begin_time = time.time()
    for epoch in range(0, num_of_epochs):
        ###### TRAIN
        train_accu = []
        train_loss = []
        _model.train()
        start_time = time.time()
        print("Epoch: {}".format(epoch))
        random.shuffle(base_dirs)
        for base_dir in base_dirs:
            print('Using {}'.format(base_dir))
            d = dataset.get_data_loader(base_dir, img_type, batch_size=util.BATCH_SIZE)
            if optimizer_type == 'adam':
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if ('step' in state and state['step'] >= 1024):
                            state['step'] = 1000
            for idx, (img, label, err) in enumerate(d):
                if(sum(err) != 0):
                    print("Error occured")
                    continue
                # ipdb.set_trace()
                label = label.cuda()
                img = Variable(img.float()).cuda().contiguous()
                img = img.view(img.shape[0], 1, img.shape[1], img.shape[2], img.shape[3])
                # print(img.shape)
                out = _model(img)
                # print(out)
                optimizer.zero_grad()
                loss = criterion(out, label)

                loss.backward()
                optimizer.step()

                prediction = out.data.max(1)[1]
                correct = float(prediction.eq(label.data).sum())
                accuracy = (correct / float(util.BATCH_SIZE)) * 100.0
                train_loss.append(loss.item())
                train_accu.append(accuracy)
                # if count % 100 == 0:
                print("Epoch {} iter {}: Training accuracy = {}/{}={} Loss = [{}]".format(epoch, idx, correct, util.BATCH_SIZE, accuracy, loss.item()))
                count+= 1
        torch.save(_model, './{}/{}'.format(save_dir, '{}.ckpt'.format(epoch)))
    train_accu = np.asarray(train_accu)
    train_loss = np.asarray(train_loss)
    np.save(file=os.path.join(save_dir, 'loss.npy'), arr=train_loss)
    np.save(file=os.path.join(save_dir, 'train_acc.npy'), arr=train_accu)
    print("Ran for {}s".format(time.time()-begin_time))

if __name__ == '__main__':
    run(optimizer_type='adam', img_type=util.ImgType.STRUCTURAL_T1)
    # run_test()
