"""
  Filename    [ statistics.py ]
  PackageName [ DLCVSpring2019 - YOLOv1 ]
  Synposis    [ Statistic Information of dataset ]
"""

import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import dataset
import models
import predict
import utils
import visualize_bbox


def draw_bbox():
    """ Draw the boundary box with the textfile message"""
    images = ["0076", "0086", "0907"]
    
    for img in images:
        imgpath = os.path.join("hw2_train_val", "val1500", "images", img+".jpg")
        detpath = os.path.join("hw2_train_val", "val1500", "labelTxt_hbb_pred", img+".txt")
        outpath = img + ".jpg"

        visualize_bbox.visualize(imgpath, detpath, outpath)

def count_class(dataloader: DataLoader):
    """ 
    Count the time of appreance of each class in the Dataset 

    Parameters
    ----------
    dataloader : DataLoader
        The dataset to be count

    Return
    ------
    labels : list

    counts : list

    """
    labelEncoder = utils.labelEncoder
    labels = labelEncoder.inverse_transform(
        torch.linspace(0, 15, steps=16).type(torch.long).unsqueeze(-1)
    )
    counts = torch.zeros(16, dtype=torch.long)

    for index, (_, target, _) in enumerate(trainLoader, 1):
        class_onehot = target[:, :, :, 10:].type(torch.long)
        count = class_onehot.sum(0).sum(0).sum(0)
        
        counts += count

    labels = labels.data.tolist()

    return labels, counts.data.tolist()

def main():
    """ HW2 Question 6 """
    trainset = dataset.MyDataset(
        root="hw2_train_val/train15000", 
        grid_num=14, 
        train=False, 
        transform=transforms.ToTensor()
    )
    trainLoader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    
    labels, counts = count_class(trainLoader)

    with open("dataset_information", "w") as textfile:
        textfile.write(counts)

if __name__ == "__main__":
    main()