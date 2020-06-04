"""
  Filename     [ dataset.py ]
  PackageName  [ DLCV Spring 2019 - YOLOv1 ]
  Synposis     [ DataLoader of the aerial dataset ]
"""

import os
import random
import sys
import time

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import DataLoader, Dataset

import utils

__all__ = ['MyDataset']

class MyDataset(Dataset):
    classnames = utils.classnames
    labelEncoder = utils.labelEncoder
    oneHotEncoder = utils.oneHotEncoder

    def __init__(self, root, grid_num=7, bbox_num=2, class_num=16, train=True, transform=None):
        """ Save the imageNames and the labelNames. """
        self.filenames = []
        self.train     = train
        self.transform = transform
        self.grid_num  = grid_num
        self.bbox_num  = bbox_num
        self.class_num = class_num

        image_folder = os.path.join(root, "images")
        anno_folder  = os.path.join(root, "labelTxt_hbb")

        imageNames = os.listdir(image_folder)

        for name in imageNames:
            imageName = os.path.join(image_folder, name)
            labelName = os.path.join(anno_folder, name.split(".")[0] + ".txt")
            
            self.filenames.append((imageName, labelName))
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        imageName, labelName = self.filenames[index]
        
        image = Image.open(imageName)
        boxes, classIndexs = self.readtxt(labelName)
        
        if self.train:
            image = self.RandomAdjustHSV(image, 0.95, 1.05)
            if random.random() < 0.5: image, boxes = self.HorizontalFlip(image, boxes)
            if random.random() < 0.5: image, boxes = self.VerticalFlip(image, boxes)
            
        target = self.encoder(boxes, classIndexs, image.size)
        target = torch.from_numpy(target)
        
        if self.transform: 
            image = self.transform(image)

        return image, target, labelName

    def encoder(self, boxes, classindex, image_size):
        """
        Parameters
        ----------
        boxes : numpy.array
            [N, 4], contains [x1, y1, x2, y2] in integers
        
        labels : numpy.array
            [N, self.class_num]
        
        Return
        ------
        targets : numpy.array
            [self.grid_num, self.grid_num, self.class_num]
        """
        image_size = np.asarray(image_size)
        image_size = np.concatenate((image_size, image_size), axis=0)

        target    = np.zeros((self.grid_num, self.grid_num, 5 * self.bbox_num + self.class_num))
        boxes     = boxes / image_size
        cell_size = 1. / self.grid_num
        wh        = boxes[:, 2:] - boxes[:, :2]
        centerXY  = (boxes[:, 2:] + boxes[:, :2]) / 2
       
        ij = (np.ceil(centerXY / cell_size) - 1).astype(int)
        
        # Confidence
        for index, (i, j) in enumerate(ij):
            # print("Index: {}, i: {}, j: {}".format(index, i, j))
            target[j, i] = 0    # Reset as zero
            
            target[j, i, 4] = 1
            target[j, i, 9] = 1
            target[j, i, classindex + 10] = 1

            # Coordinate transform to xyhw
            cornerXY = ij[index] * cell_size
            deltaXY  = (centerXY[index] - cornerXY) / cell_size
            target[j, i, 2:4] = wh[index]
            target[j, i,  :2] = deltaXY
            target[j, i, 7:9] = wh[index]
            target[j, i, 5:7] = deltaXY

        # Target in numpy
        return target

    def readtxt(self, labelName):
        """ 
        Transfer the labels to the tensor. 

        Parameters
        ----------
        labelName: str
            the label textfile to open
        
        Return
        ------
        target: np.array
            [7 * 7 * 26]
        """
        with open(labelName, "r") as textfile:
            labels = textfile.readlines()
            labels = np.asarray("".join(labels).replace("\n", " ").strip().split()).reshape(-1, 10)

        classNames  = np.asarray(labels[:, 8])
        classIndexs = self.labelEncoder.transform(classNames)
        
        boxes = np.asarray(labels[:, :8]).astype(np.float)
        boxes = np.concatenate((boxes[:, :2], boxes[:, 4:6]), axis=1)

        return boxes, classIndexs

    def RandomAdjustHSV(self, img, min_f, max_f, prob=0.5):
        """ Augmentation Method: Adjust HSV """
        if random.random() < prob:
            factor = random.uniform(min_f, max_f)
            choice = random.randint(0, 3)
            
            if choice == 0:
                img = ImageEnhance.Color(img).enhance(factor)            
            elif choice == 1:
                img = ImageEnhance.Brightness(img).enhance(factor)
            elif choice == 2:
                img = ImageEnhance.Contrast(img).enhance(factor)
            elif choice == 3:
                img = ImageEnhance.Sharpness(img).enhance(factor)

        return img

    def HorizontalFlip(self, im, boxes):
        """ Augmentation Method: Horizontal Flip """
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
        h, w = im.size
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 0] = xmin
        boxes[:, 2] = xmax

        return im, boxes

    def VerticalFlip(self, im, boxes):
        """ Augmentation Method: Vertical Flip """
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        h, w = im.size
        ymin = h - boxes[:, 3]
        ymax = h - boxes[:, 1]
        boxes[:, 1] = ymin
        boxes[:, 3] = ymax

        return im, boxes

    def ZoomIn(self, im, boxes, scale):
        """ Augmentation Method: Zoom In (Not Suggest) """
        h, w = im.size
        boundary = int(w * (scale - 1) / 2)

        im = im.resize((int(h * scale), int(w * scale)), Image.ANTIALIAS)
        im = im.crop((boundary, boundary, boundary + h, boundary + w))

        boxes = (boxes * scale - boundary).astype(int).clip(min=0, max=w)

        return im, boxes

class Testset(Dataset):
    def __init__(self, img_root, grid_num=7, bbox_num=2, class_num=16, transform=None):
        """ Save the imageNames and the labelNames and read in future. """
        self.filenames = [ os.path.join(img_root, name) for name in os.listdir(img_root) ]
        self.transform = transform
        self.grid_num  = grid_num
        self.bbox_num  = bbox_num
        self.class_num = class_num

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        if self.transform: 
            image = self.transform(image)
        
        return image, self.filenames[index].split(".")[0]

def main():
    return

if __name__ == "__main__":
    main()
