"""
  Filename    [ statistics.py ]
  PackageName [ DLCVSpring2019 - YOLOv1 ]
  Synposis    [ Statistic Information of dataset ]
"""

import os
from collections import Counter

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import dataset
import models
import predict
import utils
from visualize_bbox import visualize


def draw_bbox():
    """ Draw the boundary box with the textfile message"""    
    for img in ("0076", "0086", "0907"):
        imgpath = os.path.join("hw2_train_val", "val1500", "images", img + ".jpg")
        detpath = os.path.join("hw2_train_val", "val1500", "labelTxt_hbb_pred", img + ".txt")
        outpath = img + ".jpg"

        visualize(imgpath, detpath, outpath)

# TODO: Rewrite the function using textReader
def count_class(detpath):
    """ 
    Count the number of bbox for each categories in dataset

    Parameters
    ----------
    detpath : str

    Return
    ------
    counter : Counter
    """
    counter = Counter()
    detfiles = os.listdir(detpath)

    for idx, fname in enumerate(detfiles, 1):
        print("Predicted: [{}/{} ({:.2%})]\r".format(
            idx, len(detfiles), idx / len(detfiles)), end=""
        )
        with open(os.path.join(detpath, fname), "r") as textfile:
            labels = "".join(textfile.readlines()).replace("\n", " ").strip().split()
            labels = np.asarray(labels).reshape(-1, 10)

            # Instance Counter
            classNames  = np.asarray(labels[:, 8])
            counter.update(classNames)
            
            # TODO: Size Statistics
            boxes = np.asarray(labels[:, :8]).astype(np.float)
            boxes = np.concatenate((boxes[:, :2], boxes[:, 4:6]), axis=1)

    return counter

def main():
    """ HW2 Question 6 """
    counter = count_class(os.path.join("hw2_train_val", "train15000", "labelTxt_hbb"))
    print(counter)

if __name__ == "__main__":
    main()
