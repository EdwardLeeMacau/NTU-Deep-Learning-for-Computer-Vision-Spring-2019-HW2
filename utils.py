import numpy az np

class_Label = {}

def labelToTensor(labels):
    """ Copy parse_gt from eval and vbbox. """
    pass
   
def tensorToLabel(tensors):
    pass

def IoU(boxA, boxB):
    x_1 = max(boxA[0], boxB[0])
    y_1 = max(boxA[1], boxB[1])
    x_2 = min(boxA[2], boxB[2])
    y_2 = min(boxA[3], boxB[3])
    
    intersectionArea = max(0, x_2 - x_1) * max(0, y_2 - y_1)
    unionArea = 0
    iou = 0
    
    return

def nonMaximumSupression():
    pass

class counter:
    def __init__(self):
        """ Copy from AIML """
        pass
