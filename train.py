import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import models
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.filenames = []
        self.root      = root
        self.transform = transform

        filenames = glob.glob(os.path.join(root, "*.png"))

        raise NotImplementedError # TODO: Read the image and label.

        self.len = len(self.filenames)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image_filename, label = self.filenames[index]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size = 5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fullconnect1 = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fullconnect2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 320)
        x = self.fullconnect1(x)
        x = self.fullconnect2(x)
        
        return x

def create_MNISTLoader(trainingSet, testingSet, show=False):
    trainingSet_loader = DataLoader(trainingSet, batch_size=64, shuffle=True, num_workers=1)
    testingSet_loader  = DataLoader(testingSet, batch_size=1000, shuffle=False, num_workers=1)

    return trainingSet_loader, testingSet_loader

def imshow(img):
    np_Image = img.numpy()
    plt.imshow(np.transpose(np_Image, (1, 2, 0)))

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
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(data), len(traindataloader.dataset), 100. * batch_idx / len(traindataloader), loss.item()))
            
            if save_interval > 0:
                if (iteration % save_interval == 0) and (iteration > 0):
                    saveCheckpoint("MNIST-{}.pth".format(iteration), model, optimizer)

            iteration += 1

        test(model, valdataloader, device)
    
    if save_interval > 0:
        saveCheckpoint("MNIST-{}.pth".format(iteration), model, optimizer)

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
    print("\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss, correct, len(testdataloader.dataset), 100. * correct / len(testdataloader.dataset)))

def saveCheckpoint(checkpoint_path, model: nn.Module, optimizer):
    state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, checkpoint_path)

    print("Model saved to {}".format(checkpoint_path))

def loadCheckpoint(checkpoint_path, model: nn.Module, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])

    print('Model loaded from {}'.format(checkpoint_path))

    return model, optimizer

def main():
    # trainingSet, testingSet = create_MNISTSet(show=True)
    # trainingSet_loader, testingSet_loader = create_MNISTLoader(trainingSet, testingSet)

    # """ Show MNIST Batch """
    # dataIter = iter(trainingSet_loader)
    # images, labels = dataIter.next()
    # print("Image tensor in each batch: {}, {}".format(images.shape, images.dtype))
    # print("Label tensor in each batch: {}, {}".format(labels.shape, labels.dtype))

    # imshow(torchvision.utils.make_grid(images))
    # print('Labels:')
    # print(' '.join('%5s' % labels[j] for j in range(16)))

    # device = selectDevice(show=True)

    # """ Create CNN """
    # model = Net().to(device)
    # print(model)

    # """ Train the CNN """
    # model = train(model, trainingSet_loader, testingSet_loader, 5, device, log_interval=100, save_interval=500)

    # "" Fine-tune """
    # print(model.state_dict().keys())
    
    trainset = MyDataset(root="hw2_train_val/train15000", transform=transforms.ToTensor())
    testset  = MyDataset(root="hw2_train_val/test1500", transform=transforms.ToTensor())

    device = selectDevice(show=True)
    model  = models.Yolov1_vgg16bn(pretrained=True)

    model = train(model, trainset_loader, testset_loader, 5, device, log_interval=100, save_interval=500)

if __name__ == "__main__":
    main()