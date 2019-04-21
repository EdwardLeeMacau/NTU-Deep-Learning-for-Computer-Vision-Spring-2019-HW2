import os
import dataset
import visualize_bbox
import predict
import models

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms

import utils

labelEncoder = utils.labelEncoder
trainset = dataset.MyDataset(root="hw2_train_val/train15000", size=15000, grid_num=14, train=False, transform=transforms.ToTensor())
trainLoader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

def draw_bbox():
    images = ["0076", "0086", "0907"]
    
    for img in images:
        imgpath = os.path.join("hw2_train_val", "val1500", "images", img+".jpg")
        detpath = os.path.join("hw2_train_val", "val1500", "labelTxt_hbb_pred", img+".txt")
        outpath = img + ".jpg"

        visualize_bbox.visualize(imgpath, detpath, outpath)

def count_class(dataloader: DataLoader):
    labels = labelEncoder.inverse_transform(torch.linspace(0, 15, steps=16).type(torch.long).unsqueeze(-1))
    counts = torch.zeros(16, dtype=torch.long)

    # print(labels)
    # raise Error

    for index, (_, target, _) in enumerate(trainLoader, 1):
        print(index * 64)
        class_onehot = target[:, :, :, 10:].type(torch.long)
        count = class_onehot.sum(0).sum(0).sum(0)
        
        print(count.data.tolist())
        counts += count

    print(counts)
    labels = labels.data.tolist()
    return labels, counts.data.tolist()

if __name__ == "__main__":
    # draw_bbox()

    labels, counts = count_class(trainLoader)
    print(labels)
    print(counts)

    with open("dataset_information", "w") as textfile:
        textfile.write(counts)
