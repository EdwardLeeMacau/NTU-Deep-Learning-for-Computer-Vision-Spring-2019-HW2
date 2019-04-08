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
import logging
import logging.config
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import skimage

import models
import utils
import dataset

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)

def selectDevice(show=False):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device used: ", device)

    return device

def train(model, traindataloader, valdataloader, epochs, device, log_interval=100, save_interval=500, save=True):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()

    iteration = 0
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(traindataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            optimizer.step()

            if (iteration % log_interval == 0):
                logger.info("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch, batch_idx * len(data), len(traindataloader.dataset), 100. * batch_idx / len(traindataloader), loss.item()))
            if save_interval > 0:
                if (iteration % save_interval == 0) and (iteration > 0):
                    utils.saveCheckpoint("Yolov1-{}.pth".format(iteration), model, optimizer)

            iteration += 1

        test(model, valdataloader, device)
    
    if save_interval > 0:
        utils.saveCheckpoint("Yolov1-{}.pth".format(iteration), model, optimizer)

    return model

def test(model, testdataloader: DataLoader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in testdataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(testdataloader.dataset)
    logger.info("\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, len(testdataloader.dataset), 100. * correct / len(testdataloader.dataset)))

def main():    
    trainset = dataset.MyDataset(root="hw2_train_val/train15000", size=15000, transform=transforms.ToTensor())
    testset  = dataset.MyDataset(root="hw2_train_val/test1500", size=1500, transform=transforms.ToTensor())

    trainset_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)
    testset_loader  = DataLoader(testset, batch_size=1500, shuffle=False, num_workers=1)

    device = selectDevice(show=True)
    model  = models.Yolov1_vgg16bn(pretrained=True)

    model = train(model, trainset_loader, testset_loader, 5, device, log_interval=100, save_interval=500)

    test(model, testset_loader, device)

if __name__ == "__main__":
    main()