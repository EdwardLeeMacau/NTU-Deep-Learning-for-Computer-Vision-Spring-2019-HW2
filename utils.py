import numpy as np
import torch
from torch import nn
from torch import optim

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

classnames = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 
    'harbor', 'swimming-pool', 'helicopter', 'container-crane'
]

labelEncoder  = LabelEncoder().fit(classnames)
oneHotEncoder = OneHotEncoder(sparse=False).fit(labelEncoder.transform(classnames).reshape(16, 1))

def selectDevice(show=False):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device used: ", device)

    return device

def saveCheckpoint(checkpoint_path, model, optimizer, scheduler: optim.lr_scheduler.MultiStepLR, epoch):
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'epoch': epoch + 1,
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

def loadModel(checkpoint_path: str, model: nn.Module):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer'])
    print('Model loaded from %s' % checkpoint_path)

    return model

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