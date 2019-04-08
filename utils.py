import numpy as np
from torch import nn

class_Label = {}

def imshow(img):
    np_Image = img.numpy()
    plt.imshow(np.transpose(np_Image, (1, 2, 0)))

def labelToTensor(labels):
    objects = []

    """
      object_struct = {
          "name": ...
          "difficult": ...
          "bbox": ...
      }
    """
    
    # line.strip().split()

    for label in labels:
        object_struct = {}
        object_struct['name'] = label[8]
        
        # if len(label) != 10: continue

        if (len(label) == 9):
            object_struct['difficult'] = 0
        elif (len(label) == 10):
            object_struct['difficult'] = int(label[9])

        # prob = float(label[9])
        
        object_struct['bbox'] = [int(float(label[0])),
                                 int(float(label[1])),
                                 int(float(label[4])),
                                 int(float(label[5]))]
        
        w = int(float(label[4])) - int(float(label[0]))
        h = int(float(label[5])) - int(float(label[1]))
        object_struct['area'] = w * h
        
        objects.append(object_struct)

    raise NotImplementedError

    return objects
   
def tensorToLabel(tensors):
    raise NotImplementedError

def IoU(boxA, boxB):
    x_1 = max(boxA[0], boxB[0])
    y_1 = max(boxA[1], boxB[1])
    x_2 = min(boxA[2], boxB[2])
    y_2 = min(boxA[3], boxB[3])
    
    intersectionArea = max(0, x_2 - x_1) * max(0, y_2 - y_1)
    unionArea = [ (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]) 
                + (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]) 
                - intersectionArea ]

    iou = intersectionArea / unionArea
    
    return iou

def IoU_xywh(boxA: np.array, boxB: np.array):
    raise NotImplementedError

def nonMaximumSupression(basicThreshold, bboxs):
    
    
    raise NotImplementedError
    return bboxs

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

class counter:
    def __init__(self):
        """ Copy from AIML """
        pass
