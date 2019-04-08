import os
import sys
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

# import cv2
import skimage
# import data_augment
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
              'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
              'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 
              'harbor', 'swimming-pool', 'helicopter', 'container-crane']

classname2index = {}
index2classname = {}

for index, classname in enumerate(classnames, 1):
    classname2index[classname] = index
    index2classname[index] = classname

class MyDataset(Dataset):
    image_size = 448

    def __init__(self, root, size, grid_num=7, class_num=26, train=True, transform=None):
        """ Save the imageNames only but transfer the labels as the tensor. """
        self.filenames = []
        self.root      = root
        self.train     = train
        self.transform = transform
        self.grid_num  = grid_num
        self.class_num = class_num

        numDigits = len(str(size))
        for i in range(0, size):
            i = str(i).zfill(numDigits)
            
            imageName = i + ".png"
            labelName = i + ".txt"
            imageName = os.path.join(root, "images", imageName)
            labelName = os.path.join(root, "labelTxt_hbb", labelName)
            
            self.filenames.append((imageName, labelName))
            # self.filenames.append((imageName, target))
        
        self.len = len(self.filenames)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        imageName, labelName = self.filenames[index]
        image       = skimage.io.imread(imageName)
        img_resize  = None
        target      = self.label2target(labelName, image.size)
        # width, height = image.width, height
        
        return img_resize, target

    def encoder(self, boxes, labels, image_size):
        """
        Args:
          boxes:    [N, 4], contains [x1, y1, x2, y2] in integers
          labels:   [N, self.class_num]
        
        Return:
          targets:  [self.grid_num, self.grid_num, self.class_num]
        """
        image_size = torch.tensor(image_size, dtype=torch.float)
        image_size = torch.cat((image_size, image_size), 0)
        print(image_size.shape)

        target    = torch.zeros((self.grid_num, self.grid_num, self.class_num))
        boxes     = boxes / image_size
        cell_size = 1. / self.grid_num
        wh        = boxes[:, 2:] - boxes[:, :2]
        centerXY  = (boxes[:, 2:] + boxes[:, :2]) / 2
        
        for cxcy_sample, index in enumerate(centerXY):
            ij = (cxcy_sample / cell_size).ceil() - 1
            i, j = int(ij[0]), int(ij[1])

            # Confidence, class one-hot encoding
            target[j, i, 4] = 1
            target[j, i, 9] = 1
            target[j, i, labels[index] + 9] = 1
            
            # Coordinate transform to xyhw
            xy = ij * cell_size
            delta_xy = (cxcy_sample - xy) / cell_size
            target[j, i, 2:4] = wh[index]
            target[j, i,  :2] = delta_xy
            target[j, i, 7:9] = wh[index]
            target[j, i, 5:7] = delta_xy

        return target

    def label2target(self, labelName, image_size):
        target = torch.zeros(self.grid_num, self.grid_num, self.class_num)

        with open(labelName, "r") as textfile:
            labels = textfile.readlines()

            boxes = torch.zeros(len(labels), 4)
            classIndexs = []
            # difficultys = []

            for index, label in enumerate(labels):
                if len(label) == 0: continue
                
                label = label.strip().split()
                # print(label)

                corner      = [float(element) for element in label[0:2] + label[4:6]]
                className   = label[8]
                classIndex  = classname2index[className]
                # difficulty  = int(label[9]) 

                boxes[index] = torch.tensor(corner)
                classIndexs.append(classIndex)
                # difficultys.append(difficulty)
            
            # print(boxes)
            target = self.encoder(boxes, classIndexs, image_size)

        return target

def dataset_unittest():
    trainset = MyDataset(root="hw2_train_val/train15000", size=15000, transform=transforms.ToTensor())
    testset  = MyDataset(root="hw2_train_val/test1500", train=False, size=1500, transform=transforms.ToTensor())

    trainset_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)
    testset_loader  = DataLoader(testset, batch_size=1500, shuffle=False, num_workers=1)
    
    # Make loader iterator
    train_iter = iter(trainset_loader)
    for i in range(0, 100):
        img, target = next(train_iter)
        print(img, target)

if __name__ == '__main__':
    dataset_unittest()
