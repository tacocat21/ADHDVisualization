import numpy as np
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist
import torchvision
import dataset

import resnet_3d

import h5py
import cv2

from multiprocessing import Pool


IMAGE_SIZE = 224
NUM_CLASSES = 101
batch_size = 25
lr = 0.0001
num_of_epochs = 10


def run():
    model = resnet_3d.resnet50(sample_size=IMAGE_SIZE, sample_duration=16)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
    structuralImages = dataset.ImageDataset()

    for epoch in range(0, num_of_epochs):

        ###### TRAIN
        train_accu = []
        model.train()
        random_indices = np.random.permutation(len(train[0]))
        start_time = time.time()
        for i in range(0, len(train[0]) - batch_size, batch_size):

            augment = True
            video_list = [(train[0][k], augment)
                          for k in random_indices[i:(batch_size + i)]]
            data = pool_threads.map(loadSequence, video_list)

            next_batch = 0
            for video in data:
                if video.size == 0:  # there was an exception, skip this
                    next_batch = 1
            if (next_batch == 1):
                continue

            x = np.asarray(data, dtype=np.float32)
            x = Variable(torch.FloatTensor(x), requires_grad=False).cuda().contiguous()

            y = train[1][random_indices[i:(batch_size + i)]]
            y = torch.from_numpy(y).cuda()

            with torch.no_grad():
                h = model.conv1(x)
                h = model.bn1(h)
                h = model.relu(h)
                h = model.maxpool(h)

                h = model.layer1(h)
                h = model.layer2(h)
                h = model.layer3(h)
            h = model.layer4[0](h)

            h = model.avgpool(h)

            h = h.view(h.size(0), -1)
            output = model.fc(h)

            # output = model(x)

            loss = criterion(output, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            prediction = output.data.max(1)[1]
            accuracy = (float(prediction.eq(y.data).sum()) / float(batch_size)) * 100.0
            if (epoch == 0):
                print(i, accuracy)
            train_accu.append(accuracy)
        accuracy_epoch = np.mean(train_accu)
        print(epoch, accuracy_epoch, time.time() - start_time)

        ##### TEST
        model.eval()
        test_accu = []
        random_indices = np.random.permutation(len(test[0]))
        t1 = time.time()
        for i in range(0, len(test[0]) - batch_size, batch_size):
            augment = False
            video_list = [(test[0][k], augment)
                          for k in random_indices[i:(batch_size + i)]]
            data = pool_threads.map(loadSequence, video_list)

            next_batch = 0
            for video in data:
                if video.size == 0:  # there was an exception, skip this batch
                    next_batch = 1
            if (next_batch == 1):
                continue

            x = np.asarray(data, dtype=np.float32)
            x = Variable(torch.FloatTensor(x)).cuda().contiguous()

            y = test[1][random_indices[i:(batch_size + i)]]
            y = torch.from_numpy(y).cuda()

            # with torch.no_grad():
            #     output = model(x)
            with torch.no_grad():
                h = model.conv1(x)
                h = model.bn1(h)
                h = model.relu(h)
                h = model.maxpool(h)

                h = model.layer1(h)
                h = model.layer2(h)
                h = model.layer3(h)
                h = model.layer4[0](h)
                # h = model.layer4[1](h)

                h = model.avgpool(h)

                h = h.view(h.size(0), -1)
                output = model.fc(h)

            prediction = output.data.max(1)[1]
            accuracy = (float(prediction.eq(y.data).sum()) / float(batch_size)) * 100.0
            test_accu.append(accuracy)
            accuracy_test = np.mean(test_accu)

        print('Testing', accuracy_test, time.time() - t1)

    torch.save(model, '3d_resnet.model')
    pool_threads.close()
    pool_threads.terminate()

def run_test():
    model = torch.load('3d_resnet.model')
    model.cuda()

    ##### save predictions directory
    prediction_directory = 'UCF-101-predictions-3d/'
    if not os.path.exists(prediction_directory):
        os.makedirs(prediction_directory)
    for label in class_list:
        if not os.path.exists(prediction_directory + label + '/'):
            os.makedirs(prediction_directory + label + '/')

    acc_top1 = 0.0
    acc_top5 = 0.0
    acc_top10 = 0.0
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
    random_indices = np.random.permutation(len(test[0]))
    mean = np.asarray([0.485, 0.456, 0.406], np.float32)
    std = np.asarray([0.229, 0.224, 0.225], np.float32)
    model.eval()

    for i in range(len(test[0])):

        t1 = time.time()

        index = random_indices[i]

        filename = test[0][index]
        filename = filename.replace('.avi', '.hdf5')
        filename = filename.replace('UCF-101', 'UCF-101-hdf5')

        h = h5py.File(filename, 'r')
        nFrames = len(h['video'])

        data = np.zeros((nFrames, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        for j in range(nFrames):
            frame = h['video'][j]
            frame = frame.astype(np.float32)
            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
            frame = frame / 255.0
            frame = (frame - mean) / std
            frame = frame.transpose(2, 0, 1)
            data[j, :, :, :] = frame
        h.close()

        num_frames = 16
        loop_i = list(range(0, nFrames, num_frames)) # get 16 frames
        prediction = np.zeros((len(loop_i) -1, NUM_CLASSES), dtype=np.float32)

        # loop_i.append(nFrames - num_frames)
        # TODO: change here to make it 3d
        batch_size = 25

        for k in range(0, len(loop_i) - 1, batch_size):
            data_batch = []
            for j in range(k, min(len(loop_i) - 1, k+batch_size)):
                _d = data[loop_i[j]:loop_i[j + 1]]
                _d = _d.transpose(1, 0, 2, 3)
                data_batch.append(_d)

            # data_batch = []
            # for i in range(loop_i[j], loop_i[j+1]): # do no overlap
            #     _d = np.asarray(data[i:i+num_frames], dtype=np.float32)
            #
            #     _d = torch.FloatTensor(_d)
            #     _d.view(3, num_frames, IMAGE_SIZE, IMAGE_SIZE) # change to transpose
            #     data_batch.append(np.array(_d))

            # forward
            with torch.no_grad():
                x = np.asarray(data_batch, dtype=np.float32)
                x = Variable(torch.FloatTensor(x)).cuda().contiguous()
                # output = model(x)
                # x = x.view(x.shape[0], 3, num_frames, IMAGE_SIZE, IMAGE_SIZE)
                h = model.conv1(x)
                h = model.bn1(h)
                h = model.relu(h)
                h = model.maxpool(h)

                h = model.layer1(h)
                h = model.layer2(h)
                h = model.layer3(h)
                h = model.layer4[0](h)

                h = model.avgpool(h)

                h = h.view(h.size(0), -1)
                output = model.fc(h)

            # prediction[loop_i[j]:loop_i[j + 1]] = output.cpu().numpy()
            prediction[k: min(k+batch_size, len(loop_i)-1)] = output.cpu().numpy()
        filename = filename.replace(data_directory + 'UCF-101-hdf5/', prediction_directory)
        if (not os.path.isfile(filename)):
            with h5py.File(filename, 'w') as h:
                h.create_dataset('predictions', data=prediction)

        # softmax
        for j in range(prediction.shape[0]):
            prediction[j] = np.exp(prediction[j]) / np.sum(np.exp(prediction[j]))

        prediction = np.sum(np.log(prediction), axis=0)
        argsort_pred = np.argsort(-prediction)[0:10]

        label = test[1][index]
        confusion_matrix[label, argsort_pred[0]] += 1
        if (label == argsort_pred[0]):
            acc_top1 += 1.0
        if (np.any(argsort_pred[0:5] == label)):
            acc_top5 += 1.0
        if (np.any(argsort_pred[:] == label)):
            acc_top10 += 1.0

        print('i:%d nFrames:%d t:%f (%f,%f,%f)'
              % (i, nFrames, time.time() - t1, acc_top1 / (i + 1), acc_top5 / (i + 1), acc_top10 / (i + 1)))

    number_of_examples = np.sum(confusion_matrix, axis=1)
    for i in range(NUM_CLASSES):
        confusion_matrix[i, :] = confusion_matrix[i, :] / np.sum(confusion_matrix[i, :])

    results = np.diag(confusion_matrix)
    indices = np.argsort(results)

    sorted_list = np.asarray(class_list)
    sorted_list = sorted_list[indices]
    sorted_results = results[indices]

    for i in range(NUM_CLASSES):
        print(sorted_list[i], sorted_results[i], number_of_examples[indices[i]])

    np.save('multiple_frame_confusion_matrix.npy', confusion_matrix)



if __name__ == '__main__':
    # run()
    run_test()