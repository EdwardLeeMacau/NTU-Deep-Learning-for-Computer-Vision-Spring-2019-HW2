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
import random 
# import argparse
import pdb
import logging
import logging.config
from PIL import Image

import numpy as np

import models
from dataset import MyDataset
import utils
import cmdparse

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)

classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 
            'harbor', 'swimming-pool', 'helicopter', 'container-crane']

labelEncoder  = LabelEncoder()
oneHotEncoder = OneHotEncoder(sparse=False)
integerEncoded = labelEncoder.fit_transform(classnames)
oneHotEncoded  = oneHotEncoder.fit_transform(integerEncoded.reshape(16, 1))

def decode(output: torch.Tensor, prob_min=0.05, iou_threshold=0.5, grid_num=7, bbox_num=2, class_num=16):
    """
    Args:
      output: [batch_size, grid_num, grid_num, 5 * bbox_num + class_num]
    
    Return:
      keep_boxes: <list of list>
      classNames: <list of list>
    """
    grid_num = 7
    probs = []
    boxes = []
    classIndexs = []
    cell_size   = 1. / grid_num
    batch_size  = output.shape[0]
    
    output = output.data
    output = output.squeeze(0) # [7, 7, 26]
    # print("Output.shape: {}".format(output.shape))
    # print("Output: {}".format(output))

    contain1 = output[:, :, 4].unsqueeze(-1)
    contain2 = output[:, :, 9].unsqueeze(-1)
    # print("Contain1.shape: {}".format(contain1.shape))
    contain = torch.cat((contain1, contain2), -1)
    # print("Contain.shape: {}".format(contain.shape))
    # print(contain[3, 3])
    
    mask1 = (contain > prob_min)
    mask2 = (contain == contain.max()) #we always select the best contain_prob what ever it>0.9
    mask  = (mask1 + mask2).gt(0)
    # print(mask[3, 3])
    # print("Mask.shape: {}".format(mask.shape))
    
    # TODO: redifine xy: the coordinate of each cell's up-left corner  
    # TODO: redifine box_xy: [x1 y1 x2 y2] related to the image

    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i, j, b] == 1:
                    box = output[i, j, b * 5: b * 5 + 4]
                    contain_prob = output[i, j, b*5+4].type(torch.float)
                    # contain_prob = torch.FloatTensor([output[i, j, b * 5 + 4]])
                        
                    # Recover the base of xy as image_size
                    xy = torch.tensor([j, i], dtype=torch.float).cuda().unsqueeze(0) * cell_size
                    # print("xy.shape: {}".format(xy.shape))
                    # print("xy: {}".format(xy))

                    box[:2] = box[:2] * cell_size + xy
                    box_xy  = torch.zeros(box.size(), dtype=torch.float)
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]                        
                    max_prob, classIndex = torch.max(output[i, j, 10:], 0)
                    # print("classIndex: {}".format(classIndex))
                    # print("max_prob.shape: {}".format(max_prob.shape))
                    # print("max_prob: {}".format(max_prob))

                    if float((contain_prob * max_prob).item()) > prob_min:
                        classIndex = classIndex.unsqueeze(0)
                        boxes.append(box_xy.view(1, 4))
                        classIndexs.append(classIndex)
                        probs.append((contain_prob * max_prob).view(1))

    if len(boxes) == 0:
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        classIndexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes, 0) #(n,4)
        probs = torch.cat(probs, 0) #(n,)
        classIndexs = torch.cat(classIndexs, 0) #(n,)
    
    if random.random() < 0.1:
        print("*** Show random answer: ")
        print("*** Boxes: {}".format(boxes))
        print("*** Probs: {}".format(probs))
        print("*** ClassIndex: {}".format(classIndexs))

    keep_index = nonMaximumSupression(boxes, probs, iou_threshold)

    return boxes[keep_index], classIndexs[keep_index], probs[keep_index]

def nonMaximumSupression(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold):
    """
    Not generalize to multi-img processing, only 1 in 1 out.

    Args:
      boxes:  [N, 4], (x1, y1, x2, y2)
      scores: [N]
    
    Return:
      keep_boxes: [x]
    """    
    _, index = scores.sort(descending=True)
    # print(index)
    keep_boxes = []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    # 1 image case first
    while index.numel() > 0:
        # print(index)
        # print(index[0].item())
        # i = index[0].item()
        # keep_boxes.append(i)

        # Check if it is the last bbox: break
        if index.numel() == 1:  
            keep_boxes.append(index.item())
            break
        
        i = index[0].item()
        keep_boxes.append(i)
        
        # Check index runs well
        # print("x1[index[1:]] {}".format(x1[index[1:]]))

        # IoU calculating
        xx1 = x1[index[1:]].clamp(min=x1[i])
        yy1 = y1[index[1:]].clamp(min=y1[i])
        xx2 = x2[index[1:]].clamp(max=x2[i])
        yy2 = y2[index[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w*h

        # print("IoU: {}".format(IoU(boxes[i], boxes[index[1: ]])))
        ovr = inter / (areas[i] + areas[index[1:]] - inter)
        # print("Ovr: {}".format(ovr))
        # Supress the bbox where overlap area > iou_threshold, return the remain index
        ids = (ovr <= iou_threshold).nonzero().squeeze()
        # print(ids)
        # IoU calculated.

        # print("ids.shape: {}".format(ids.shape))
        # print("ids: {}".format(ids))

        # Check if it is no bbox remains: break
        if ids.numel() == 0: break
        index = index[ids + 1]

    return torch.tensor(keep_boxes, dtype=torch.long)

def IoU(box: torch.Tensor, remains: torch.Tensor):
    """
    Calcuate the IoU of the specific bbox and other boxes.

    Args:
      box:     [5]
      remains: [num_remain, 5]
    
    Return:
      iou: [num_remain]
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

"""
def predict(images: torch.Tensor, model):
    output = model(images)
    boxes, classIndexs, probs = decode(output, prob_min=0.05, iou_threshold=0.5, grid_num=7, bbox_num=2)

    return boxes, classIndexs, probs
"""

def export(boxes, classNames, probs, labelName, outputpath="hw2_train_val/val1500/labelTxt_hbb_pred", image_size=512.):
    """ Write one output file with the boxes and the classnames. """
    boxes = (boxes * image_size).round()
    rect  = torch.zeros(boxes.shape[0], 8)

    # Extand (x1, y1, x2, y2) to (x1, y1, x2, y1, x2, y2, x1, y2)
    rect[:,  :3] = boxes[:, :3]
    rect[:, 4:6] = boxes[:, 2:]
    rect[:, 3]   = boxes[:, 1]
    rect[:, 6]   = boxes[:, 0]
    rect[:, 7]   = boxes[:, 3]

    # Return the probs to string lists
    round_func = lambda x: round(x, 3)
    probs = list(map(str, list(map(round_func, probs.data.tolist()))))
    classNames = list(map(str, classNames))

    with open(os.path.join(outputpath, labelName.split("/")[-1]), "w") as textfile:
        for i in range(0, rect.shape[0]):
            prob = probs[i]
            className = classNames[i]

            textfile.write(" ".join(map(str, rect[i].data.tolist())) + " ")
            textfile.write(" ".join((className, prob)) + "\n")

def decode_unittest():
    output = torch.zeros(1, 7, 7, 26)
    target = torch.zeros_like(output)

    obj   = torch.tensor([0.5, 0.5, 0.2, 0.8, 1])
    classIndex = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
    target[:, 3, 3] = torch.cat((obj, obj, classIndex), dim=0)
    output[:, 3, 3] = torch.cat((obj, obj, classIndex), dim=0)

    boxes, classIndexs, probs = decode(output, prob_min=0.05, iou_threshold=0.5, grid_num=7, bbox_num=2)
    classNames = labelEncoder.inverse_transform(classIndexs.type(torch.long).to("cpu"))

def main():
    """
    Workflow:
    1.  Image to tensors
    2.  Predict form tensors
        2.1 Supress the bbox that doesn't contain object (by prob_min)
        2.2 Execute NMS (by nonMaximumSupression)
    """
    os.system("clear")
    start = time.time()

    torch.set_default_dtype(torch.float)
    device = utils.selectDevice()

    model = models.Yolov1_vgg16bn(pretrained=True).to(device)
    model = utils.loadModel(cmdparse.args.model, model)
    print("Read Model: {}".format(cmdparse.args.model))
    
    trainset  = MyDataset(root="hw2_train_val/train15000", train=False, size=15000, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]))

    testset  = MyDataset(root="hw2_train_val/val1500", train=False, size=1500, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]))

    trainset_loader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4)
    testset_loader  = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    # Return the imageName for storing the predict_msg
    if not os.path.exists(cmdparse.args.output):
        os.mkdir(cmdparse.args.output)

    if not os.path.exists("hw2_train_val/train15000/labelTxt_hbb_pred"):
        os.mkdir("hw2_train_val/train15000/labelTxt_hbb_pred")

    # Testset prediction
    for data, target, labelName in testset_loader:
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        boxes, classIndexs, probs = decode(output, prob_min=0.05, iou_threshold=0.5, grid_num=7, bbox_num=2)
        
        classNames = labelEncoder.inverse_transform(classIndexs.type(torch.long).to("cpu"))

        # Write the output file
        if cmdparse.args.export: 
            export(boxes, classNames, probs, labelName[0])
            logger.info("Wrote file: {}".format(labelName[0].split("/")[-1]))

    # Trainset prediction
    """
    for data, _, labelName in trainset_loader:
        data = data.to(device)

        output = model(data)
        boxes, classIndexs, probs = decode(output, prob_min=0.05, iou_threshold=0.5, grid_num=7, bbox_num=2)

        classNames = labelEncoder.inverse_transform(classIndexs.type(torch.long).to("cpu"))

        # Write the output file
        if cmdparse.args.export:
            export(boxes, classNames, probs, labelName[0], outputpath="hw2_train_val/train15000/labelTxt_hbb_pred")
            logger.info("Wrote file: {}".format(labelName[0].split("/")[-1]))
    """
        
    end = time.time()
    logger.info("Used Time: {} min {:.0f} s".format((end - start) // 60, (end - start) % 60))

if __name__ == "__main__":
    # decode_unittest()
    main()
