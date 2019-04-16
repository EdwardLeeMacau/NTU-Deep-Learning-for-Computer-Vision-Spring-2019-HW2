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

def train(model, train_dataloader, val_dataloader, epochs, device, lr=0.001, log_interval=100, save_interval=500, save=True):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = models.YoloLoss(7, 2, 5, 0.5, device).to(device)
    model.train()

    iteration = 0
    loss_list = []
    mean_aps  = []

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 50], gamma=0.1)

    for epoch in range(1, epochs + 1):
        model.train()
        scheduler.step()

        for batch_idx, (data, target, _) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            optimizer.step()

            if (iteration % log_interval == 0):
                logger.info("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch, batch_idx * len(data), len(train_dataloader.dataset), 100. * batch_idx / len(train_dataloader), loss.item()))
            
            iteration += 1

        val_loss = test_loss(model, criterion, val_dataloader, device)
        loss_list.append(val_loss)
        
        if epoch > 20:
            mean_ap = test_map(model, criterion, val_dataloader, device)
            mean_aps.append((epoch, mean_ap))
        
        if epoch == 2:
            utils.saveCheckpoint("Yolov1-{}.pth".format(epoch), model, optimizer)
        
        if epoch == 10:
            utils.saveCheckpoint("Yolov1-{}.pth".format(epoch), model, optimizer)
        
        if (epoch >= 20) and (epoch % 5 == 0):
            utils.saveCheckpoint("Yolov1-{}.pth".format(epoch), model, optimizer)

    with open("Training_Record.txt", "w") as textfile:
        textfile.write("Loss: {}".format(str(loss_list)))
        textfile.write("\n")
        textfile.write("Mean_aps: {}".format(str(mean_aps)))

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
    logger.info("*** Test set - Average loss: {:.4f}".format(loss))

    return loss

def test_map(model, criterion, dataloader: DataLoader, device):
    model.eval()
    mean_ap = 0

    # Calculate the map value
    with torch.no_grad():
        for data, target, labelNames in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            boxes, classIndexs, probs = predict.decode(output, prob_min=0.1, iou_threshold=0.5, grid_num=7, bbox_num=2)
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
    model = models.Yolov1_vgg16bn(pretrained=True).to(device)
    model = train(model, trainLoader, testLoader, args.epochs, device, lr=args.lr, log_interval=10, save_interval=0)

    end = time.time()
    logger.info("*** Training ended.")
    logger.info("Used Time: {} hours {} min {:.0f} s".format((end - start) // 3600, (end - start) // 60, (end - start) % 60))

if __name__ == "__main__":
    os.system("clear")

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, help="Reload the model")
    subparsers = parser.add_subparsers(required=True, dest="command")

    basic_parser = subparsers.add_parser("basic")
    basic_parser.add_argument("--lr", default=0.001, type=float, help="Set the initial learning rate")
    basic_parser.add_argument("--batchs", default=16, type=int, help="Set the epochs")
    basic_parser.add_argument("--epochs", default=50, type=int, help="Set the epochs")
    basic_parser.add_argument("--worker", default=4, type=int, help="Set the workers")
    
    improve_parser = subparsers.add_parser("improve")
    improve_parser.add_argument("--lr", default=0.001, type=float, help="Set the initial learning rate")
    improve_parser.add_argument("--batchs", default=16, type=int, help="Set the epochs")
    improve_parser.add_argument("--epochs", default=50, type=int, help="Set the epochs")
    improve_parser.add_argument("--worker", default=4, type=int, help="Set the workers")
    
    args = parser.parse_args()
    print(args)

    main()
