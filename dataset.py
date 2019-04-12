import os
import sys
import os.path
import time

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import PIL.Image as Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# import skimage
# import data_augment
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
              'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
              'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 
              'harbor', 'swimming-pool', 'helicopter', 'container-crane']

    labelEncoder  = LabelEncoder()
    oneHotEncoder = OneHotEncoder(sparse=False)
    integerEncoded = labelEncoder.fit_transform(classnames)
    oneHotEncoded  = oneHotEncoder.fit_transform(integerEncoded.reshape(16, 1))

    def __init__(self, root, size, grid_num=7, bbox_num=2, class_num=16, train=True, transform=None):
        """ Save the imageNames only but transfer the labels as the tensor. """
        self.filenames = []
        self.root      = root
        self.train     = train
        self.transform = transform
        self.grid_num  = grid_num
        self.bbox_num  = bbox_num
        self.class_num = class_num

        numDigits = len(str(size))
        for i in range(0, size):
            i = str(i).zfill(numDigits)
            
            imageName = i + ".jpg"
            labelName = i + ".txt"
            imageName = os.path.join(root, "images", imageName)
            labelName = os.path.join(root, "labelTxt_hbb", labelName)
            
            self.filenames.append((imageName, labelName))
        
        self.len = len(self.filenames)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        imageName, labelName = self.filenames[index]
        
        image    = Image.open(imageName)
        target   = self.label2target(labelName, image.size)
        if self.transform: image = self.transform(image)

        return image, target

    def encoder(self, boxes, classindex, image_size):
        """
        Args:
          boxes:    [N, 4], contains [x1, y1, x2, y2] in integers
          labels:   [N, self.class_num]
        
        Return:
          targets:  [self.grid_num, self.grid_num, self.class_num]
        """
        image_size = np.asarray(image_size)
        image_size = np.concatenate((image_size, image_size), axis=0)

        target    = np.zeros((self.grid_num, self.grid_num, 5 * self.bbox_num + self.class_num))
        boxes     = boxes / image_size
        cell_size = 1. / self.grid_num
        wh        = boxes[:, 2:] - boxes[:, :2]
        centerXY  = (boxes[:, 2:] + boxes[:, :2]) / 2
       
        ij = (np.ceil(centerXY / cell_size) - 1).astype(int)
        i, j = ij[:, 0], ij[:, 1]
        # print(i, j)
        # print(i.shape, j.shape)
        # print(target[j, :, :])
        # print(target[j, i, 4])
        # print(target[j, i, 4].shape)

        # Confidence
        target[j, i, 4] = 1
        target[j, i, 9] = 1
        target[j, i, classindex + 9] = 1

        # Coordinate transform to xyhw
        cornerXY = ij * cell_size
        deltaXY  = (centerXY - cornerXY) / cell_size
        target[j, i, 2:4] = wh
        target[j, i,  :2] = deltaXY
        target[j, i, 7:9] = wh
        target[j, i, 5:7] = deltaXY

        # for index, cxcy_sample in enumerate(centerXY):
        #     ij = torch.ceil(cxcy_sample / cell_size) - 1
        #     # ij = (cxcy_sample / cell_size).ceil() - 1
        #     i, j = int(ij[0]), int(ij[1])

        #     # Confidence, class one-hot encoding
        #     target[j, i, 4] = 1
        #     target[j, i, 9] = 1
            
            
            # Coordinate transform to xyhw
        #     xy = ij * cell_size
        #     delta_xy = (cxcy_sample - xy) / cell_size
        #     target[j, i, 2:4] = wh[index]
        #     target[j, i,  :2] = delta_xy
        #     target[j, i, 7:9] = wh[index]
        #     target[j, i, 5:7] = delta_xy

        target = torch.from_numpy(target)

        # print("target.shape: {}".format(target.shape))

        return target

    def label2target(self, labelName, image_size):
        """ Transfer the labels to the tensor. """
        with open(labelName, "r") as textfile:
            labels = textfile.readlines()

            labels      = np.asarray("".join(labels).replace("\n", " ").strip().split()).reshape(-1, 10)
            classNames  = np.asarray(labels[:, 8])
            classIndexs = self.labelEncoder.transform(classNames)
            
            boxes = np.asarray(labels[:, :8]).astype(np.float)
            boxes = np.concatenate((boxes[:, :2], boxes[:, 4:6]), axis=1)

        target = self.encoder(boxes, classIndexs, image_size)

        return target

def dataset_unittest():
    trainset = MyDataset(root="hw2_train_val/train15000", size=15000, transform=transforms.Compose([
        transforms.Resize((448, 448)), 
        transforms.ToTensor()
    ]))

    testset  = MyDataset(root="hw2_train_val/test1500", train=False, size=1500, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]))

    trainset_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    testset_loader  = DataLoader(testset, batch_size=1500, shuffle=False, num_workers=4)
    
    # Make loader iterator
    train_iter = iter(trainset_loader)

    start = time.time()
    for i in range(0, 64):
        img, target = next(train_iter)
        # print("target: {}".format(target))

    print("Using time: {}".format(time.time() - start))

if __name__ == '__main__':
    dataset_unittest()
