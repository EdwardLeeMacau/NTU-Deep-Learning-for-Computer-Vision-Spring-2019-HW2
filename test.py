import torch
import models
import os
import sys

def val():
    pass

def test():
    pass

def IoU():
    pass

def nonMaximumSupression(bboxs):
    basicThreshold = 0.1
    pass

def main():
    imagepath = sys.argv[1]
    detpath = sys.argv[2]
    annopath = sys.argv[3]
    modelpath = sys.argv[4]
    
    if not os.path.exists(detpath):
        os.mkdir(detpath)
    
    model = models.Yolov1_vgg16bn(pretrained=True)
    pass
    
if __name__ == "__main__":
    main()
