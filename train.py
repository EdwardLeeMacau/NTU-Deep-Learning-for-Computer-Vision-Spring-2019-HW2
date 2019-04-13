import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

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

parser = argparse.ArgumentParser()
parser.add_argument("--worker", default=4, type=int)
args = parser.parse_args()

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)

def selectDevice(show=False):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device used: ", device)

    return device

def train(model, traindataloader, valdataloader, epochs, device, log_interval=100, save_interval=500, save=True):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = models.YoloLoss(7, 2, 5, 0.5, device).to(device)
    model.train()

    iteration = 0
    loss_list = []
    accuracy_list = []
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(traindataloader):
            target = target.type(torch.cuda.FloatTensor)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            optimizer.step()

            if (iteration % log_interval == 0):
                logger.info("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch, batch_idx * len(data), len(traindataloader.dataset), 100. * batch_idx / len(traindataloader), loss.item()))
            
            # if save_interval > 0:
            #     if (iteration % save_interval == 0) and (iteration > 0):
            #         logger.info("Save Checkpoint: {}".format("Yolov1-{}-{}.pth".format(epoch, iteration)))
            #         utils.saveCheckpoint("Yolov1-{}-{}.pth".format(epoch, iteration), model, optimizer)

            iteration += 1

        test_loss = test(model, valdataloader, device)
        loss_list.append(test_loss)
       
        if epoch % 3 == 0:
            utils.saveCheckpoint("Yolov1-{}.pth".format(epoch), model, optimizer)

    with open("Training_Record.txt", "w") as textfile:
        textfile.write("\n".join(loss))
        textfile.write("\n")
        textfile.write("\n".join(accuracy))    

    return model

def test(model, testdataloader: DataLoader, device):
    criterion = models.YoloLoss(7, 7, 5, 0.5, device).to(device)
    model.eval()
    test_loss = 0
    # correct = 0

    with torch.no_grad():
        for data, target in testdataloader:
            target = target.type(torch.cuda.FloatTensor)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            # prediction = output.max(1, keepdim=True)
            # print("Prediction.shape: {}".format(prediction.shape))
            # print("Target.shape: {}".format(target.shape))
            # correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(testdataloader.dataset)
    # logger.info("\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
    #             test_loss, correct, len(testdataloader.dataset), 100. * correct / len(testdataloader.dataset)))
    logger.info("\n Test set: Average loss: {:.4f}\n".format(test_loss))

    return test_loss

def train_test_unittest():    
    # Load dataset
    trainset = dataset.MyDataset(root="hw2_train_val/train15000", size=15000, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]))
    testset  = dataset.MyDataset(root="hw2_train_val/test1500", size=1500, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]))

    trainLoader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=args.worker)
    testLoader  = DataLoader(testset, batch_size=64, shuffle=False, num_workers=args.worker)

    device = selectDevice(show=True)
    model  = models.Yolov1_vgg16bn(pretrained=True).to(device)

    # Train the model
    model = train(model, trainLoader, testLoader, 1, device, log_interval=10, save_interval=0)

    # Test the model
    test(model, testLoader, device)

def main():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    trainset = dataset.MyDataset(root="hw2_train_val/train15000", size=15000, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]))

    testset  = dataset.MyDataset(root="hw2_train_val/val1500", size=1500, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]))

    trainLoader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=args.worker)
    testLoader  = DataLoader(testset, batch_size=64, shuffle=False, num_workers=args.worker)

    device = selectDevice(show=True)
    model  = models.Yolov1_vgg16bn(pretrained=True).to(device)

    model = train(model, trainLoader, testLoader, 30, device, log_interval=10, save_interval=0)

if __name__ == "__main__":
    # train_test_unittest()
    main()
