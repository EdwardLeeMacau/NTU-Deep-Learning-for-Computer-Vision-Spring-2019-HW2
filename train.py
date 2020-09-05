"""
  Filename     [ train.py ]
  PackageName  [ DLCV Spring 2019 - YOLOv1 ]
  Synposis     [ Training Procedure of YOLOv1 ]
"""

import argparse
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import DataLoader, Dataset

import dataset
import evaluation
import models
import inference
import utils

classnames    = utils.classnames
labelEncoder  = utils.labelEncoder
oneHotEncoder = utils.oneHotEncoder

# TODO: Train Backbone / Train the Dense Layer only.
def train(model, criterion, optimizer, scheduler, train_loader, val_loader, 
          start_epochs, epochs, device, grid_num=7, lr=0.001, log_interval=10, 
          save_name="Yolov1"):
    model.train()
    
    epochs_list     = []
    train_loss_list = []
    val_loss_list   = []
    val_mean_aps    = []

    for epoch in range(start_epochs + 1, epochs + 1):
        model.train()
        
        if scheduler: 
            scheduler.step()

        iteration = 0
        train_loss = 0

        # Train and backpropagation
        for batch_idx, (data, target, _) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (iteration % log_interval == 0):
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
            iteration += 1
        
        train_loss /= iteration
        val_loss = test_loss(model, criterion, val_loader, device)
        
        epochs_list.append(epoch)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        print("*** Train set - Average loss: {:.4f}".format(train_loss))
        print("*** Test set - Average loss: {:.4f}".format(val_loss))
        
        if epoch > 0:
            mean_ap = test_map(model, criterion, val_loader, device, grid_num=14)
            val_mean_aps.append(mean_ap)
            
            if mean_ap >= max(val_mean_aps):
                utils.saveCheckpoint(save_name + "-{}.pth".format(epoch), model, optimizer, scheduler, epoch)
        else:
            val_mean_aps.append(0)

        plt.clf()
        plt.plot(epochs_list, train_loss_list, label="Training loss")
        plt.plot(epochs_list, val_loss_list, label="Validation loss")
        plt.legend(loc=0)
        plt.title("Loss vs Epochs")
        plt.savefig("loss.png")
        plt.clf()
        
        plt.plot(epochs_list, val_mean_aps, label="mAP")
        plt.legend(loc=0)
        plt.title("mAP vs Epochs")
        plt.savefig("mAP.png")

    return model 
    
def test_loss(model, criterion, dataloader: DataLoader, device):
    model.eval()
    loss = 0

    with torch.no_grad():
        for data, target, _ in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()

    loss /= len(dataloader.dataset)

    return loss

def test_map(model, criterion, dataloader: DataLoader, device, grid_num):
    """ 
    Evaluate the performance of model by mAP (mean Average Percision) 

    Parameters
    ----------
    model : torch.nn.Module

    loader : torch.utils.data.DataLoader

    grid_num : 

    Return
    ------
    mAP :
        mAP score
    """
    model.eval()
    mean_ap = 0

    # Calculate the map value
    with torch.no_grad():
        for data, target, labelNames in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            boxes, classIndexs, probs = predict.decode(output, prob_min=args.prob, iou_threshold=args.iou, grid_num=grid_num, bbox_num=2)
            classNames = labelEncoder.inverse_transform(classIndexs.type(torch.long).to("cpu"))
            predict.export(boxes, classNames, probs, labelNames[0], outputpath="hw2_train_val/val1500/labelTxt_hbb_pred")
        
        classaps, mean_ap = evaluation.scan_map(
            detpath="hw2_train_val/val1500/labelTxt_hbb_pred/",
            annopath="hw2_train_val/val1500/labelTxt_hbb/"
        )

    return mean_ap

def main():
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    grid_num = 7 if args.command == "basic" else 14

    trainset = dataset.MyDataset(
        root="hw2_train_val/train15000", 
        grid_num=grid_num,
        train=args.augment, 
        transform=transform
    )

    testset  = dataset.MyDataset(
        grid_num=grid_num,
        root="hw2_train_val/val1500", 
        train=False, 
        transform=transform
    )

    trainLoader = DataLoader(trainset, batch_size=args.batchs, shuffle=True, num_workers=args.worker)
    testLoader  = DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.worker)
    device = utils.selectDevice(show=True)

    if args.command == "basic":
        model = models.Yolov1_vgg16bn(pretrained=True).to(device)
        criterion = models.YoloLoss(7., 2., 5., 0.5, device).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 45, 55], gamma=0.1)
        start_epoch = 0

        if args.load:
            model, optimizer, start_epoch, scheduler = utils.loadCheckpoint(args.load, model, optimizer, scheduler)

        model = train(model, criterion, optimizer, scheduler, trainLoader, testLoader, start_epoch, args.epochs, device, lr=args.lr, grid_num=7)

    elif args.command == "improve":
        model_improve = models.Yolov1_vgg16bn_Improve(pretrained=True).to(device)
        criterion = models.YoloLoss(14., 2., 5, 0.5, device).to(device)
        optimizer = optim.SGD(model_improve.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 40, 70], gamma=0.1)
        start_epoch = 0
        
        if args.load:
            model_improve, optimizer, start_epoch, scheduler = utils.loadCheckpoint(args.load, model, optimizer, scheduler)

        model_improve = train(model_improve, criterion, optimizer, scheduler, trainLoader, testLoader, start_epoch, args.epochs, device, lr=args.lr, grid_num=7, save_name="Yolov1-Improve")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, help="Reload the model")
    parser.add_argument("--augment", action="store_true", help="Open the augment function")
    parser.add_argument("--iou", type=float, default=0.5, help="Lower Bound of IOU value")
    parser.add_argument("--prob", type=float, default=0.1, help="Lower Bound of Prob_obj")
    subparsers = parser.add_subparsers(required=True, dest="command")

    basic_parser = subparsers.add_parser("basic")
    basic_parser.add_argument("--lr", default=1e-3, type=float, help="Set the initial learning rate")
    basic_parser.add_argument("--batchs", default=16, type=int, help="Set the epochs")
    basic_parser.add_argument("--epochs", default=60, type=int, help="Set the epochs")
    basic_parser.add_argument("--worker", default=4, type=int, help="Set the workers")
    
    improve_parser = subparsers.add_parser("improve")
    improve_parser.add_argument("--lr", default=1e-3, type=float, help="Set the initial learning rate")
    improve_parser.add_argument("--batchs", default=16, type=int, help="Set the epochs")
    improve_parser.add_argument("--epochs", default=80, type=int, help="Set the epochs")
    improve_parser.add_argument("--worker", default=4, type=int, help="Set the workers")
    
    args = parser.parse_args()

    main()
