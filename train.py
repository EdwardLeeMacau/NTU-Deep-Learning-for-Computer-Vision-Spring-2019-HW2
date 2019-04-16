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

    for epoch in range(1, epochs + 1):
        model.train()

        if epoch == 21:
            lr = 0.0002

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            logger.info("Learning rate adjusted to: {}".format(lr))

        if epoch == 41:
            lr = 0.0005
        
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            logger.info("Learning rate adjusted to: {}".format(lr))

        if epoch == 61:
            lr = 0.00001
        
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            logger.info("Learning rate adjusted to: {}".format(lr))

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

        test_loss, mean_ap = test(model, val_dataloader, device, epoch)
        loss_list.append(test_loss)
        mean_aps.append(mean_ap)
        
        if (epoch >= 20) and (epoch % 5 == 0):
            utils.saveCheckpoint("Yolov1-{}.pth".format(epoch), model, optimizer)

    with open("Training_Record.txt", "w") as textfile:
        textfile.write("Loss: {}".format(str(loss_list)))
        textfile.write("\n")
        textfile.write("Mean_aps: {}".format(str(mean_aps)))

    return model

def test(model, dataloader: DataLoader, device, epochs):
    criterion = models.YoloLoss(7, 2, 5, 0.5, device).to(device)
    model.eval()
    test_loss = 0
    mean_ap = 0

    with torch.no_grad():
        for data, target, _ in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()

    test_loss /= len(dataloader.dataset)
    logger.info("*** Test set - Average loss: {:.4f}".format(test_loss))

    # Calculate the map value
    if epochs > 20:
        for data, target, labelNames in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            boxes, classIndexs, probs = predict.decode(output, prob_min=0.1, iou_threshold=0.5, grid_num=7, bbox_num=2)
            classNames = labelEncoder.inverse_transform(classIndexs.type(torch.long).to("cpu"))
            predict.export(boxes, classNames, probs, labelNames[0], outputpath="hw2_train_val/val1500/labelTxt_hbb_pred")
        
        classaps, mean_ap = evaluation.scan_map()

        logger.info("*** Test set - MAP: {:.4f}".format(mean_ap))
        logger.info("*** Test set - AP: {}".format(classaps))
    
    return test_loss, mean_ap

def main():
    start = time.time()

    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    trainset = dataset.MyDataset(root="hw2_train_val/train15000", size=15000, train=True, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]))

    testset  = dataset.MyDataset(root="hw2_train_val/val1500", size=1500, train=False, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]))

    trainLoader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=args.worker)
    testLoader  = DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.worker)

    device = utils.selectDevice(show=True)
    model  = models.Yolov1_vgg16bn(pretrained=True).to(device)
    model  = train(model, trainLoader, testLoader, args.epochs, device, lr=args.lr, log_interval=10, save_interval=0)

    end = time.time()
    logger.info("*** Training ended.")
    logger.info("Used Time: {} hours {} min {:.0f} s".format((end - start) // 3600, (end - start) // 60, (end - start) % 60))

if __name__ == "__main__":
    os.system("clear")

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.001, type=float, help="Set the initial learning rate")
    parser.add_argument("--epochs", default=50, type=int, help="Set the epochs")
    parser.add_argument("--worker", default=4, type=int, help="Set the workers")
    args = parser.parse_args()

    main()
