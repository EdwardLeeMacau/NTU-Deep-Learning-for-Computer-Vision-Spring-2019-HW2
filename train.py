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
import predict

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

def train(model, train_dataloader, val_dataloader, epochs, device, lr=0.0001, log_interval=100, save_interval=500, save=True):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = models.YoloLoss(7, 2, 5, 0.5, device).to(device)
    model.train()

    iteration = 0
    loss_list = []
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target, _) in enumerate(train_dataloader):
            target = target.type(torch.cuda.FloatTensor)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            optimizer.step()

            if (iteration % log_interval == 0):
                logger.info("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch, batch_idx * len(data), len(train_dataloader.dataset), 100. * batch_idx / len(train_dataloader), loss.item()))
            
            # if save_interval > 0:
            #     if (iteration % save_interval == 0) and (iteration > 0):
            #         logger.info("Save Checkpoint: {}".format("Yolov1-{}-{}.pth".format(epoch, iteration)))
            #         utils.saveCheckpoint("Yolov1-{}-{}.pth".format(epoch, iteration), model, optimizer)

            iteration += 1

        test_loss, mean_average_precision = test(model, val_dataloader, device)
        loss_list.append(test_loss)
       
        if epoch % 3 == 0:
            utils.saveCheckpoint("Yolov1-{}.pth".format(epoch), model, optimizer)

    with open("Training_Record.txt", "w") as textfile:
        textfile.write("\n".join(loss_list))

    return model

def test(model, dataloader: DataLoader, device):
    criterion = models.YoloLoss(7, 7, 5, 0.5, device).to(device)
    model.eval()
    test_loss = 0
    mean_average_precision = 0

    with torch.no_grad():
        for data, target, _ in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()

    test_loss /= len(dataloader.dataset)
    logger.info("*** Test set - Average loss: {:.4f} \n".format(test_loss))
    logger.info("*** Test set - MAP: {:.4f} \n".format(mean_average_precision))

    return test_loss, mean_average_precision

def main():
    start = time.time()

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
    model  = train(model, trainLoader, testLoader, 30, device, lr=0.0001, log_interval=10, save_interval=0)

    end = time.time()
    logger.info("*** Training ended.")
    logger.info("Used Time: {} hours {} min {:.0f} s".format((end - start) // 3600, (end - start) // 60, (end - start) % 60))

    # Try to generate the textfiles.
    start = time.time()
    for data, target, _ in testLoader:
        textmsg = predict.predict(data, model)
    end = time.time()
    
    logger.info("Used Time: {} hours {} min {:.0f} s".format((end - start) // 3600, (end - start) // 60, (end - start) % 60))

if __name__ == "__main__":
    main()
