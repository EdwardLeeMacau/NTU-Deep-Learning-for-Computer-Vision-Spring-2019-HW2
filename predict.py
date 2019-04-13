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
import argparse
import pdb
import logging
import logging.config
from PIL import Image

import numpy as np

import models
from dataset import MyDataset
import utils

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="The model parameters file to read.", required=True)
parser.add_argument("--output", type=str, default="hw2_train_val/val1500/labelTxt_hbb_pred", help="The path to save the labels.", required=True)
parser.add_argument("--export", action="store_true", help="Store the results to the outputPath.")
args = parser.parse_args()

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
    # print("Output.shape: {}".format(output.shape))
    output = output.squeeze(0) # [batch_size, 7, 7, 26]
    # print("Output.shape: {}".format(output.shape))
    # print("Output: {}".format(output))
    
    contain1 = output[:, :, 4].unsqueeze(-1)
    contain2 = output[:, :, 9].unsqueeze(-1)
    # print("Contain1.shape: {}".format(contain1.shape))
    contain = torch.cat((contain1, contain2), -1)
    # print("Contain.shape: {}".format(contain.shape))
    
    mask1 = (contain > prob_min)
    mask2 = (contain == contain.max()) #we always select the best contain_prob what ever it>0.9
    mask  = (mask1 + mask2).gt(0)
    # print("Mask.shape: {}".format(mask.shape))
    
    # TODO: redifine xy: the coordinate of each cell's up-left corner  
    # TODO: redifine box_xy: [x1 y1 x2 y2] related to the image

    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i, j, b] == 1:
                    box = output[i, j, b*5:b*5+4]
                    contain_prob = torch.FloatTensor([output[i, j, b*5+4]])
                        
                    # Recover the base of xy as image_size

                    xy = torch.cuda.FloatTensor([j, i]).unsqueeze(0) * cell_size      # up-left of cell
                        
                    # print("xy.shape: {}".format(xy.shape))
                    # print("xy: {}".format(xy))
                    box[:2] = box[:2] * cell_size + xy                     # return cxcy relative to image
                    box_xy  = torch.FloatTensor(box.size())                      # convert[cx, cy, w, h] to [x1, y1, x2, y2]
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]                        
                    max_prob, classIndex = torch.max(output[i, j, 10:], 0)

                    # print("max_prob.shape: {}".format(max_prob.shape))
                    # print("max_prob: {}".format(max_prob))

                    if float((contain_prob * max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1, 4))
                        classIndexs.append(classIndex)
                        probs.append(contain_prob*max_prob)

    pdb.set_trace()

    if len(boxes) == 0:
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        classIndexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes, 0) #(n,4)
        probs = torch.cat(probs, 0) #(n,)
        classIndexs = torch.cat(classIndexs, 0) #(n,)
    
    keep_index = nonMaximumSupression(boxes, probs, iou_threshold)
    pdb.set_trace()

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
    keep_boxes = []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    # 1 image case first
    while index.numel() > 0:
        i = index[0]
        keep_boxes.append(i)

        # Check if it is the last bbox: break
        if index.numel() == 1:  break
        
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
        ids = (ovr <= iou_threshold).nonzero().squeeze()
        # IoU calculated.

        # print("ids.shape: {}".format(ids.shape))
        # print("ids: {}".format(ids))

        # Check if it is no bbox remains: break
        if ids.numel() == 0: break
        index = index[ids + 1]

    return torch.LongTensor(keep_boxes)

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

def export(boxes, classNames, probs, labelName):
    """ Write one output file with the boxes and the classnames. """
    with open(os.path.join("hw2_train_val/val1500/labelTxt_hbb_pred", labelNames.split("/")[-1]), "w") as textfile:
        for i in range(0, boxes.shape[0]):
            x1 = str(float(boxes[i][0]))
            y1 = str(float(boxes[i][1]))
            x2 = str(float(boxes[i][2]))
            y2 = str(float(boxes[i][3]))
            prob = str(round(float(boxes[i][9]), 3))

            textfile.write(" ".join((x1, y1, x2, y1, x2, y2, x1, y2, classNames[i], prob)) + "\n")

def main():
    """
    Workflow:
    1.  Image to tensors
    2.  Predict form tensors
        2.1 Supress the bbox that doesn't contain object (by prob_min)
        2.2 Execute NMS (by nonMaximumSupression)
    """
    start = time.time()

    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = utils.selectDevice()

    model = models.Yolov1_vgg16bn(pretrained=True).to(device)
    model = utils.loadModel(args.model, model)
    print("Read Model: {}".format(args.model))
    
    testset  = MyDataset(root="hw2_train_val/val1500", train=False, size=1500, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]))

    testset_loader  = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    # Return the imageName for storing the predict_msg
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    for _, (data, target, labelName) in enumerate(testset_loader):
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        boxes, classIndexs, probs = decode(output, prob_min=0.05, iou_threshold=0.5, grid_num=7, bbox_num=2)
        
        classNames = labelEncoder.inverse_transform(classIndexs.type(torch.LongTensor).to("cpu"))

        # Write the output file
        if args.export: export(boxes, classNames, probs, labelName[0])
        
    end = time.time()
    logger.info("Used Time: {} min {:.0f} s".format((end - start) // 60, (end - start) % 60))

if __name__ == "__main__":
    main()
