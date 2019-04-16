import os
import dataset
import argparse

from torch.utils.data import Dataset, DataLoader
import torch

import utils

labelEncoder = utils.labelEncoder
name2index = labelEncoder.get_params()
print(name2index)

trainset = dataset.MyDataset(root="hw2_train_val/train15000", size=15000, train=False, transform=None)
trainLoader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

def main():
    counts = torch.zeros(16, dtype=torch.long)

    for _, target, _ in trainLoader:
        class_onehot = target[:, :, :, 10:].type(torch.long)
        count = class_onehot.sum(0).sum(0).sum(0)
        print(count.shape)

        counts += count

    print(counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # main()