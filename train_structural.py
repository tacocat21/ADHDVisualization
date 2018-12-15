"""
Code used to train and test the models
"""
import collections
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

from multiprocessing import Pool


IMAGE_SIZE = 224
NUM_CLASSES = 4

num_of_epochs = 150


def run(optimizer_type, img_type, lr = 0.005, large=False):
    if large:
        _model = model.StructuralModel3DFullImageLarge()
        _model_dir = 'model_large'
    else:
        _model = model.StructuralModel3DFullImage()
        _model_dir = 'model'
    _model = _model.cuda()
    if optimizer_type == 'adam':
        optimizer = optim.Adam(_model.parameters(), lr=lr)
    save_dir = '{}/{}/{}/{}'.format(_model_dir, str(img_type), optimizer_type, lr)
    util.mkdir(save_dir)
    base_dirs = ['KKI', 'NeuroIMAGE', 'OHSU', 'Peking_1', 'Peking_2', 'Peking_3', 'Pittsburgh','WashU']
    criterion = nn.CrossEntropyLoss()
    count = 0
    begin_time = time.time()
    train_acc_dict = {}
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
            d = dataset.get_data_loader(base_dir, img_type, batch_size=util.BATCH_SIZE, train=True)
            if optimizer_type == 'adam':
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if ('step' in state and state['step'] >= 1024):
                            state['step'] = 1000
            num_correct= 0
            num_total = 0
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
                num_correct += correct
                num_total += len(label)
                accuracy = (correct / float(len(label))) * 100.0
                train_loss.append(loss.item())
                train_accu.append(accuracy)
                # if count % 100 == 0:
                print("Epoch {} iter {}: Training accuracy = {}/{}={} Loss = [{}]".format(epoch, idx, correct, len(label), accuracy, loss.item()))
                count+= 1
            if num_total != 0:
                base_train_acc = float(num_correct)/num_total * 100.0
                print("Epoch {} {}: Training accuracy = {}/{}={}".format(epoch, base_dir, num_correct, num_total, base_train_acc))
                train_acc_dict[base_dir] = base_train_acc

        torch.save(_model, './{}/{}'.format(save_dir, '{}.ckpt'.format(epoch)))
    train_accu = np.asarray(train_accu)
    train_loss = np.asarray(train_loss)
    np.save(file=os.path.join(save_dir, 'loss.npy'), arr=train_loss)
    np.save(file=os.path.join(save_dir, 'train_acc.npy'), arr=train_accu)
    print("Ran for {}s".format(time.time()-begin_time))
    util.save_data(filename=os.path.join(save_dir, 'train_dict.pckl'), d=train_acc_dict)

def test(model_filename, img_type, dir_name):
    print("Testing {}".format(model_filename))
    model = torch.load(os.path.join(dir_name, model_filename))
    base_dirs = ['KKI', 'NeuroIMAGE', 'OHSU', 'Peking_1', 'Peking_2', 'Peking_3', 'Pittsburgh','WashU']
    model.eval()
    total_attempts = 0
    total_correct = 0
    res = collections.defaultdict(dict)
    for base_dir in base_dirs:
        d = dataset.get_data_loader(base_dir, img_type, batch_size=util.BATCH_SIZE, train=False)
        dataset_correct = 0
        dataset_attempt = 0
        for idx, (img, label, err) in enumerate(d):
            err_exists = sum(err) != 0
            if err_exists:
                print("Error in loading {}".format(base_dir))
                continue
            label = label.cuda()
            img = Variable(img.float()).cuda().contiguous()
            img = img.view(img.shape[0], 1, img.shape[1], img.shape[2], img.shape[3])
            # print(img.shape)
            with torch.no_grad():
                out = model(img)
            prediction = out.data.max(1)[1]
            correct = float(prediction.eq(label.data).sum())
            dataset_attempt += len(label)
            dataset_correct += correct
        if dataset_attempt != 0:
            accuracy = (dataset_correct / float(dataset_attempt)) * 100.0
            res[base_dir]['correct'] = dataset_correct
            res[base_dir]['attempt'] = dataset_attempt
            res[base_dir]['accuracy'] = accuracy
            total_correct += dataset_correct
            total_attempts += dataset_attempt
            print('Test Accuracy {}: {}/{}={}'.format(base_dir, dataset_correct, dataset_attempt, accuracy))
    res['summary']['correct'] = total_correct
    res['summary']['attempt'] = total_attempts
    res['summary']['accuracy'] = float(total_correct)/total_attempts * 100.0
    util.save_data(res, os.path.join(dir_name, 'test_result.pckl'))
    print('Total Accuracy: {}'.format(res['summary']['accuracy']))
    return res


if __name__ == '__main__':
    img_type = util.ImgType.STRUCTURAL_T1
    lr_list = [0.0001, 0.001, 0.05]
    for lr in lr_list:
        dir_name = 'model/ImgType.STRUCTURAL_T1/adam/{}/'.format(lr)
        models = range(0, 29)
        for m in models:
            try:
                test('{}.ckpt'.format(m), img_type, dir_name)
            except:
                print("error in {} {}".format(dir_name, m))
                continue
