"""
  Filename    [ utlis.py ]
  PackageName [ YOLOv1 ]
  Synposis    [ Utility function of YOLOv1 ]
"""

import sys

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch import nn, optim

classnames = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 
    'harbor', 'swimming-pool', 'helicopter', 'container-crane'
]

labelEncoder  = LabelEncoder().fit(classnames)
oneHotEncoder = OneHotEncoder(sparse=False).fit(labelEncoder.transform(classnames).reshape(16, 1))

def set_optimizer_lr(optimizer, lr):
    # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer

def selectDevice():
    """ Check use GPU or CPU """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return device

def saveCheckpoint(checkpoint_path, model, optimizer, scheduler: optim.lr_scheduler.MultiStepLR, epoch):
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'epoch': epoch,
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def loadCheckpoint(checkpoint_path: str, model: nn.Module, optimizer: optim, scheduler: optim.lr_scheduler.MultiStepLR):
    state = torch.load(checkpoint_path)
    resume_epoch = state['epoch']
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    print('model loaded from %s' % checkpoint_path)

    return model, optimizer, resume_epoch, scheduler

def saveModel(checkpoint_path: str, model: nn.Module):
    state = {
        'state_dict': model.state_dict(),
    }
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def loadModel(checkpoint_path: str, model: nn.Module):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('Model loaded from %s' % checkpoint_path)

    return model

def checkpointToModel(checkpoint_path: str, model_path: str):
    state = torch.load(checkpoint_path)

    newState = {
        'state_dict': state['state_dict']
    }

    torch.save(newState, model_path)
    
# TODO: Compare with model.YoloLoss()
def IoU(box: torch.Tensor, remains: torch.Tensor):
    """
    Calcuate the IoU of the specific bbox and other boxes.

    Args:
      box:     [5]
      remains: [num_remain, 5]
    
    Return:
      iou: [num_remain]
    """

    num_remain = remains.shape[0]
    box = box.expand_as(num_remain)
    
    intersectionArea = torch.zeros(num_remain)
    left_top     = torch.zeros(num_remain, 2)
    right_bottom = torch.zeros(num_remain, 2)

    left_top[:] = torch.max(
        box[:, :2],
        remains[:, :2]
    )

    right_bottom[:] = torch.min(
        box[:, 2:4],
        remains[:, 2:4]
    )

    inter_wh = right_bottom - left_top
    inter_wh[inter_wh < 0] = 0
    intersectionArea = inter_wh[:, 0] * inter_wh[:, 1]
    
    area_1 = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    area_2 = (remains[:, 2] - remains[:, 0]) * (remains[:, 3] - remains[:, 1])
    
    iou = intersectionArea / (area_1 + area_2 - intersectionArea)

    return iou

def main():
    checkpointToModel(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
