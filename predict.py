import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import os
import time
import random
import pdb
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

classnames = utils.classnames
labelEncoder  = utils.labelEncoder
oneHotEncoder = utils.oneHotEncoder

def decode(output: torch.Tensor, prob_min=0.1, iou_threshold=0.5, grid_num=7, bbox_num=2, class_num=16):
    """
    Args:
      output: [batch_size, grid_num, grid_num, 5 * bbox_num + class_num]
    
    Return:
      keep_boxes: <list of list>
      classNames: <list of list>
    """
    boxes, classIndexs, probs = [], [], []
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
    mask2 = (contain == contain.max())
    mask  = (mask1 + mask2).gt(0)
    # print(mask[3, 3])
    # print("Mask.shape: {}".format(mask.shape))

    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i, j, b] == 1:
                    box = output[i, j, b*5: b*5+4]
                    contain_prob = output[i, j, b*5+4].type(torch.float)
                        
                    # Recover the base of xy as image_size
                    xy = torch.tensor([j, i], dtype=torch.float).unsqueeze(0) * cell_size

                    box[:2] = box[:2] * cell_size + xy
                    box_xy  = torch.zeros(box.size(), dtype=torch.float)
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]                        
                    max_prob, classIndex = torch.max(output[i, j, 10:], 0)

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

    # Prevent the boxes go outside the image, so clamped the xy coordinate to 0-1
    boxes = boxes.clamp(min=0., max=1.)
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
    keep_boxes = []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    while index.numel() > 0:
        if index.numel() == 1:  
            keep_boxes.append(index.item())
            break
        
        i = index[0].item()
        keep_boxes.append(i)

        # IoU calculating
        xx1 = x1[index[1:]].clamp(min=x1[i])
        yy1 = y1[index[1:]].clamp(min=y1[i])
        xx2 = x2[index[1:]].clamp(max=x2[i])
        yy2 = y2[index[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[index[1:]] - inter)

        # Supress the bbox where overlap area > iou_threshold, return the remain index
        ids = (ovr <= iou_threshold).nonzero().squeeze()
        # IoU calculated.

        if ids.numel() == 0: break
        index = index[ids + 1]

    return torch.tensor(keep_boxes, dtype=torch.long)

def export(boxes, classNames, probs, labelName, outputpath, image_size=512.):
    """ Write one output file with the boxes and the classnames. """
    boxes = (boxes * image_size).round()
    rect  = torch.zeros(boxes.shape[0], 8)

    # Extand (x1, y1, x2, y2) to (x1, y1, x2, y1, x2, y2, x1, y2)
    rect[:,  :3] = boxes[:, :3]
    rect[:, 3:6] = boxes[:, 1:]
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

    boxes, classIndexs, probs = decode(output, prob_min=0.1, iou_threshold=0.5, grid_num=7, bbox_num=2)
    classNames = labelEncoder.inverse_transform(classIndexs.type(torch.long).to("cpu"))

def encoder_unittest():
    print("*** classnames: \n{}".format(classnames))
    
    indexs = labelEncoder.transform(classnames).reshape(-1, 1)
    print("*** indexs: \n{}".format(indexs))
    
    onehot = oneHotEncoder.transform(indexs)
    print("*** onehot: \n{}".format(onehot))
    
    reverse_index = oneHotEncoder.inverse_transform(onehot).reshape(-1)
    print("*** reverse index: \n{}".format(reverse_index))
    
    reverse_classnames = labelEncoder.inverse_transform(reverse_index.astype(int))
    print("*** reverse classnames: \n{}".format(reverse_classnames))
    
def system_unittest():
    dataset  = MyDataset(root="hw2_train_val/train15000", train=False, size=15000, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]))

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Testset prediction
    for _, target, labelName in loader:
        boxes, classIndexs, probs = decode(target, prob_min=0, iou_threshold=0.5)
        classNames = labelEncoder.inverse_transform(classIndexs.type(torch.long).to("cpu"))
        
        print("Raw Data: ")
        with open(labelName, "r") as textfile:
            content = textfile.readlines()
            print("\n".join(content))

        print("My Decoder: ")
        boxes = (boxes * 512).round()
        rect  = torch.zeros(boxes.shape[0], 8)
        rect[:,  :3] = boxes[:, :3]
        rect[:, 3:6] = boxes[:, 1:]
        rect[:, 6]   = boxes[:, 0]
        rect[:, 7]   = boxes[:, 3]
        round_func = lambda x: round(x, 3)
        probs = list(map(str, list(map(round_func, probs.data.tolist()))))
        classNames = list(map(str, classNames))
        for i in range(0, rect.shape[0]):
            prob = probs[i]
            className = classNames[i]
            print(" ".join(map(str, rect[i].data.tolist())) + " ")
            print(" ".join((className, prob)) + "\n")

        pdb.set_trace()

        


def main():
    """
    Workflow:
    1.  Image to tensors
    2.  Predict form tensors
        2.1 Supress the bbox that doesn't contain object (by prob_min)
        2.2 Execute NMS (by nonMaximumSupression)
        3.3 Return the bbox, classnames and probs
    3.  Output File if needed.
    4.  MAP calculation. 
    """
    
    start = time.time()

    torch.set_default_dtype(torch.float)
    device = utils.selectDevice()

    model = models.Yolov1_vgg16bn(pretrained=True).to(device)
    model = utils.loadModel(args.model, model)
    model.eval()
    
    trainset  = MyDataset(root="hw2_train_val/train15000", train=False, size=15000, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]))

    testset  = MyDataset(root="hw2_train_val/val1500", train=False, size=1500, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]))

    trainset_loader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=args.worker)
    testset_loader  = DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.worker)

    # Return the imageName for storing the predict_msg
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    if args.command == "train":
        loader = trainset_loader
    elif args.command == "val":
        loader = testset_loader

    # Testset prediction
    for batch_idx, (data, target, labelName) in enumerate(loader, 1):
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        boxes, classIndexs, probs = decode(output, prob_min=args.prob, iou_threshold=args.iou)
        
        classNames = labelEncoder.inverse_transform(classIndexs.type(torch.long).to("cpu"))
        print("ClassIndexs: {}".format(classIndexs))
        print("ClassNames: {}".format(classNames))

        export(boxes, classNames, probs, labelName[0], args.output)
        if batch_idx % 100 == 0:
            print(batch_idx)

    end = time.time()
    logger.info("Used Time: {} min {:.0f} s".format((end - start) // 60, (end - start) % 60))

if __name__ == "__main__":
    # decode_unittest()
    # encoder_unittest()
    
    system_unittest()

    # raise NotImplementedError

    os.system("clear")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Set the initial learning rate")
    parser.add_argument("--worker", default=4, type=int)
    parser.add_argument("--iou", default=0.5, type=float, help="NMS iou_threshold")
    parser.add_argument("--prob", default=0.1, type=float, help="NMS prob_min, pick up the bbox with the class_prob > prob_min")
    subparsers = parser.add_subparsers(required=True, dest="command")
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--output", default="hw2_train_val/train15000/labelTxt_hbb_pred")
    
    val_parser = subparsers.add_parser("val")
    val_parser.add_argument("--output", default="hw2_train_val/val1500/labelTxt_hbb_pred")

    args = parser.parse_args()

    print(args)

    main()
