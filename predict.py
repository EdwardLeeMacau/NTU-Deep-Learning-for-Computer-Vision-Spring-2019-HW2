import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import os
import time
import logging
import logging.config
from PIL import Image

import numpy as np

import models
from dataset import MyDataset
import utils

classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 
            'harbor', 'swimming-pool', 'helicopter', 'container-crane']

labelEncoder  = LabelEncoder()
oneHotEncoder = OneHotEncoder(sparse=False)
integerEncoded = labelEncoder.fit_transform(classnames)
oneHotEncoded  = oneHotEncoder.fit_transform(integerEncoded.reshape(16, 1))

def decode(output: torch.Tensor, threshold=0.05, grid_num=7, bbox_num=2, class_num=16):
    """
    Args:
      output: [batch_size, grid_num, grid_num, 5 * bbox_num + class_num]
    
    Return:
      boxes:
    """
    cell_size = 1. / grid_num
    classOneHot  = []

    confidence = torch.cat(output[:, :, :, 4], output[:, :, :, 9], dim=3)
    confidence_1 = output[:, :, :, 4]
    confidence_2 = output[:, :, :, 9]
    score_1 = output[:, :, :, 10:] * confidence_1
    score_2 = output[:, :, :, 10:] * confidence_2

    score_mask_1 = confidence_1 > threshold
    score_mask_2 = confidence_2 > threshold
    score_mask = (confidence == confidence.max())

    keep_boxes = nonMaximumSupression(boxes, score_1)

    classOneHot_mask = (classOneHot == classOneHot.max())
    classNames = oneHotEncoder.inverse_transform(classOneHot_mask)

    return keep_boxes, classNames

def nonMaximumSupression(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold=0.5):
    """
    Args:
      boxes: [batch_size, N, 4]
      scores: [batch_size, N]
    
    Return:
      keep_boxes
    """    
    batch_size = boxes.shape[0]

    _, index = scores.sort(descending=True)
    keep_boxes = []

    # 1 image case first
    while index.numel() > 0:
        i = index[0]
        keep_boxes.append(i)

        # Check if it is the last bbox: break
        if index.numel() == 1:  break
        
        xy1 = boxes[:,  :2]
        xy2 = boxes[:, 2:4]

        xx1 = x1[index[1:]].clamp(min=x1[i])
        yy1 = y1[index[1:]].clamp(min=y1[i])
        xx2 = x2[index[1:]].clamp(max=x2[i])
        yy2 = y2[index[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[index[1:]] - inter)
        ids = (ovr <= iou_threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        index = index[ids+1]

    return torch.LongTensor(keep_boxes)

def IoU(self, box:torch.Tensor, remains: torch.Tensor):
    """
    Calcuate the IoU of the specific bbox and other boxes.

    Args:
      box:     [10]
      remains: [num_remain, 10]
    
    Return:
      iou: [num_remain - 1]
    """

    num_remain = remains.shape[0]
    box = box.expand_as(num_remain)
    
    intersectionArea = torch.zeros(num_remain)
    left_top     = torch.zeros(num_remain, 2)
    right_bottom = torch.zeros(num_remain, 2)

    left_top[:] = torch.max(
        box[:, :2],
        remains[:, :2]
    )

    right_bottom[:] = torch.min(
        box[:, 2:4],
        remains[:, 2:4]
    )

    inter_wh = right_bottom - left_top
    inter_wh[inter_wh < 0] = 0
    intersectionArea = inter_wh[:, 0] * inter_wh[:, 1]
    
    area_1 = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    area_2 = (remains[:, 2] - remains[:, 0]) * (remains[:, 3] - remains[:, 1])
    
    iou = intersectionArea / (area_1 + area_2 - intersectionArea)

    return iou

def predict(output: torch.Tensor):
    boxes = []
    return boxes

def main():
    device = utils.selectDevice()
    model = models.Yolov1_vgg16bn(pretrained=True)
    
    trainset = MyDataset(root="hw2_train_val/train15000", size=15000, transform=transforms.Compose([
        transforms.Resize((448, 448)), 
        transforms.ToTensor()
    ]))

    testset  = MyDataset(root="hw2_train_val/test1500", train=False, size=1500, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]))

    trainset_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    testset_loader  = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    for batch_idx, (data, target) in enumerate(testset_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        predict(output)