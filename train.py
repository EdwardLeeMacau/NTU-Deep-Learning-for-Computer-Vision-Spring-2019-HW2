import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import glob
import os
import time
import argparse
import logging
import logging.config
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import skimage

import models
import utils
import dataset
import predict
import evaluation

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)

classnames    = utils.classnames
labelEncoder  = utils.labelEncoder
oneHotEncoder = utils.oneHotEncoder

def train(model, criterion, optimizer, scheduler, train_dataloader, val_dataloader, start_epochs, epochs, device, grid_num=7, lr=0.001, log_interval=10, save_name="Yolov1"):
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = models.YoloLoss(7, 2, 5, 0.5, device).to(device)    
    model.train()
    # iteration = 0
    
    loss_list = []
    mean_aps  = []
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40, 60], gamma=0.1)

    for epoch in range(start_epochs + 1, epochs + 1):
        model.train()
        scheduler.step()

        iteration = 0
        train_loss = 0

        for batch_idx, (data, target, _) in enumerate(train_dataloader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            train_loss += loss.item()
            
            optimizer.step()

            if (iteration % log_interval == 0):
                logger.info("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(data), len(train_dataloader.dataset), 100. * batch_idx / len(train_dataloader), loss.item()))
            iteration += 1
        
        train_loss /= iteration
        val_loss = test_loss(model, criterion, val_dataloader, device)
        loss_list.append(val_loss)
        
        logger.info("*** Train set - Average loss: {:.4f}".format(train_loss))
        logger.info("*** Test set - Average loss: {:.4f}".format(val_loss))
        
        if epoch > 20:
            mean_ap = test_map(model, criterion, val_dataloader, device, grid_num=7)
            mean_aps.append((epoch, mean_ap))
        
        if epoch == 10:
            utils.saveCheckpoint(save_name + "-{}.pth".format(epoch), model, optimizer, scheduler, epoch)
        
        if epoch == 20:
            utils.saveCheckpoint(save_name + "-{}.pth".format(epoch), model, optimizer, scheduler, epoch)
        
        if (epoch >= 30) and (epoch % 5 == 0):
            utils.saveCheckpoint(save_name + "-{}.pth".format(epoch), model, optimizer, scheduler, epoch)

    return model, loss_list, mean_aps

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
    model.eval()
    mean_ap = 0

    # Calculate the map value
    with torch.no_grad():
        for data, target, labelNames in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            boxes, classIndexs, probs = predict.decode(output, prob_min=0.1, iou_threshold=0.5, grid_num=grid_num, bbox_num=2)
            classNames = labelEncoder.inverse_transform(classIndexs.type(torch.long).to("cpu"))
            predict.export(boxes, classNames, probs, labelNames[0], outputpath="hw2_train_val/val1500/labelTxt_hbb_pred")
        
        classaps, mean_ap = evaluation.scan_map()

        logger.info("*** Test set - MAP: {:.4f}".format(mean_ap))
        logger.info("*** Test set - AP: {}".format(classaps))
    
    return mean_ap

def main():
    start = time.time()

    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    trainset = dataset.MyDataset(root="hw2_train_val/train15000", size=15000, train=True, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    testset  = dataset.MyDataset(root="hw2_train_val/val1500", size=1500, train=False, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    trainLoader = DataLoader(trainset, batch_size=args.batchs, shuffle=True, num_workers=args.worker)
    testLoader  = DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.worker)

    device = utils.selectDevice(show=True)

    if args.command == "basic":
        model = models.Yolov1_vgg16bn(pretrained=True).to(device)
        # criterion = models.YoloLoss(7, 2, 5, 0.5, device).to(device)
        criterion = models.YoloLoss_github(7, 2, 5, 0.5).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 40, 50], gamma=0.1)
        start_epoch = 0

        if args.load:
            model, optimizer, start_epoch, scheduler = utils.loadCheckpoint(args.load, model, optimizer, scheduler)

        model, loss_list, mean_aps = train(model, criterion, optimizer, scheduler, trainLoader, testLoader, start_epoch, args.epochs, device, lr=args.lr, grid_num=7)

        with open("Training_Record.txt", "a") as textfile:
            textfile.write("Basic Models")
            textfile.write("Loss: {}".format(str(loss_list)))
            textfile.write("Mean_aps: {}".format(str(mean_aps)))
            textfile.write("\n")

    elif args.command == "improve":
        model_improve = models.Yolov1_vgg16bn(pretrained=True).to(device)
        criterion = models.YoloLoss(14, 2, 5, 0.5, device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 40, 50], gamma=0.1)
        start_epoch = 0
        
        if args.load:
            model_improve, optimizer, start_epoch, scheduler = utils.loadCheckpoint(args.load, model, optimizer, scheduler)

        model_improve, loss_list, mean_aps = train(model_improve, criterion, optimizer, scheduler, trainLoader, testLoader, start_epoch, args.epochs, device, lr=args.lr, grid_num=14)

    elif args.command == "both":
        model = models.Yolov1_vgg16bn(pretrained=True).to(device)
        criterion = models.YoloLoss(7, 2, 5, 0.5, device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 40, 50], gamma=0.1)
        start_epoch = 0
        model, loss_list, mean_aps = train(model, criterion, optimizer, scheduler, trainLoader, testLoader, start_epoch, args.epochs, device, lr=args.lr, grid_num=7)

        model_improve = models.Yolov1_vgg16bn(pretrained=True).to(device)
        criterion = models.YoloLoss(14, 2, 5, 0.5, device).to(device)
        optimizer = optim.Adam(model_improve.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 40, 50], gamma=0.1)
        start_epoch = 0
        model_improve, loss_list, mean_aps = train(model_improve, criterion, optimizer, scheduler, trainLoader, testLoader, start_epoch, args.epochs, device, lr=args.lr, grid_num=14)

    end = time.time()
    logger.info("*** Training ended.")
    logger.info("Used Time: {} hours {} min {:.0f} s".format((end - start) // 3600, (end - start) // 60, (end - start) % 60))

if __name__ == "__main__":
    os.system("clear")

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, help="Reload the model")
    parser.add_argument("--iou", type=str)
    parser.add_argument("--prob", type=str)
    subparsers = parser.add_subparsers(required=True, dest="command")

    basic_parser = subparsers.add_parser("basic")
    basic_parser.add_argument("--lr", default=0.001, type=float, help="Set the initial learning rate")
    basic_parser.add_argument("--batchs", default=16, type=int, help="Set the epochs")
    basic_parser.add_argument("--epochs", default=70, type=int, help="Set the epochs")
    basic_parser.add_argument("--worker", default=4, type=int, help="Set the workers")
    
    improve_parser = subparsers.add_parser("improve")
    improve_parser.add_argument("--lr", default=0.001, type=float, help="Set the initial learning rate")
    improve_parser.add_argument("--batchs", default=16, type=int, help="Set the epochs")
    improve_parser.add_argument("--epochs", default=70, type=int, help="Set the epochs")
    improve_parser.add_argument("--worker", default=4, type=int, help="Set the workers")

    both_parser = subparsers.add_parser("both")
    both_parser.add_argument("--lr", default=0.001, type=float, help="Set the initial learning rate")
    both_parser.add_argument("--batchs", default=16, type=int, help="Set the epochs")
    both_parser.add_argument("--epochs", default=70, type=int, help="Set the epochs")
    both_parser.add_argument("--worker", default=4, type=int, help="Set the workers")
    
    args = parser.parse_args()
    print(args)

    main()
