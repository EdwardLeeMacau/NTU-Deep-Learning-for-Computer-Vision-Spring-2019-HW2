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
args = parser.parse_args()

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
      keep_boxes: <list of list>
      classNames: <list of list>
    """
    """
    cell_size   = 1. / grid_num
    classOneHot = []

    confidence = torch.cat(output[:, :, :, 4], output[:, :, :, 9], dim=3)
    confidence_1 = output[:, :, :, 4]
    confidence_2 = output[:, :, :, 9]
    score_1 = output[:, :, :, 10:] * confidence_1
    score_2 = output[:, :, :, 10:] * confidence_2

    score_mask_1 = confidence_1 > threshold
    score_mask_2 = confidence_2 > threshold
    score_mask = (confidence == confidence.max())

    for vector in output:
        keep_boxes = nonMaximumSupression(vector, score_1)

    classOneHot_mask = (classOneHot == classOneHot.max())
    classNames = oneHotEncoder.inverse_transform(classOneHot_mask)

    return keep_boxes, classNames
    """
    grid_num = 7
    probs = []
    boxes = []
    classIndexs = []
    cell_size   = 1. / grid_num
    
    output = output.data
    output = output.squeeze(0) #7x7x26
    contain1 = output[:, :, 4].unsqueeze(2)
    contain2 = output[:, :, 9].unsqueeze(2)
    contain = torch.cat((contain1, contain2), 2)
    
    mask1 = contain > 0.1 #大于阈值
    mask2 = (contain == contain.max()) #we always select the best contain_prob what ever it>0.9
    mask  = (mask1 + mask2).gt(0)
    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    
    # TODO: redifine xy: the coordinate of each cell's up-left corner  
    # TODO: redifine box_xy: [x1 y1 x2 y2] related to the image

    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i, j, b] == 1:
                    box = output[i, j, b*5:b*5+4]
                    contain_prob = torch.FloatTensor([output[i, j, b*5+4]])
                    
                    # Recover the base of xy as image_size
                    xy = torch.FloatTensor([j, i]) * cell_size      # cell左上角  up left of cell
                    box[:2] = box[:2] * cell_size + xy              # return cxcy relative to image
                    box_xy  = torch.FloatTensor(box.size())         #转换成xy形式    convert[cx,cy,w,h] to [x1,y1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    max_prob, classIndex = torch.max(output[i, j, 10:], 0)

                    if float((contain_prob * max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1,4))
                        classIndexs.append(classIndex)
                        probs.append(contain_prob*max_prob)

    if len(boxes) == 0:
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        classIndexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes, 0) #(n,4)
        probs = torch.cat(probs, 0) #(n,)
        classIndexs = torch.cat(classIndexs, 0) #(n,)
    
    keep_index = nonMaximumSupression(boxes,probs)
    return boxes[keep_index], classIndexs[keep_index], probs[keep_index]

def nonMaximumSupression(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold=0.5):
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
    print("Index.shape: {}".format(index.shape))
    print("Index: {}".format(index))
    
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
        print("x1[index[1:]] {}".format(x1[index[1:]]))

        # IoU calculating
        xx1 = x1[index[1:]].clamp(min=x1[i])
        yy1 = y1[index[1:]].clamp(min=y1[i])
        xx2 = x2[index[1:]].clamp(max=x2[i])
        yy2 = y2[index[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w*h

        print("IoU: {}".format(IoU(boxes[i], boxes[index[1: ]])))
        ovr = inter / (areas[i] + areas[index[1:]] - inter)
        print("Ovr: {}".format(ovr))
        ids = (ovr <= iou_threshold).nonzero().squeeze()
        # IoU calculated.

        print("ids.shape: {}".format(ids.shape))
        print("ids: {}".format(ids))

        # Check if it is no bbox remains: break
        if ids.numel() == 0: break
        index = index[ids+1]

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

def predict(images: torch.Tensor, model):
    output = model(images)

    boxes, labels = decode(output, threshold=0.05, grid_num=7, bbox_num=2)
    return boxes, labels

def main():
    start = time.time()

    torch.set_default_dtype(torch.cuda.FloatTensor)
    device = utils.selectDevice()

    model = models.Yolov1_vgg16bn(pretrained=True).to(device)

    # Test function, command out after used.
    modelNames = [name for name in os.lisdir("./") if name.endswith(".pth")]
    print("ModelNames: {}".format(modelNames))
    print("Read Model: {}".format(args.model))

    model = utils.loadModel(args.model, model)

    raise NotImplementedError
    
    testset  = MyDataset(root="hw2_train_val/val1500", train=False, size=1500, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]))

    testset_loader  = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    # Return the imageName for storing the predict_msg
    for _, (data, target, labelName) in enumerate(testset_loader):
        data, target = data.to(device), target.to(device)
        output = predict(data, model)
        
        predict_msg, classNames = decode(output)
        print(predict_msg)
        
        # Save the decoded msg

    end = time.time()
    logger.info("Used Time: {} min {:.0f} s".format((end - start) // 60, (end - start) % 60))

if __name__ == "__main__":
    main()