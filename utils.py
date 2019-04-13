import numpy as np
import torch
from torch import nn
from torch import optim

class_Label = {}

def selectDevice(show=False):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device used: ", device)

    return device

def saveCheckpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def loadCheckpoint(checkpoint_path: str, model: nn.Module, optimizer: optim):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

    return model, optimizer

def loadModel(checkpoint_path: str, model: nn.Module):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer'])
    print('Model loaded from %s' % checkpoint_path)

    return model