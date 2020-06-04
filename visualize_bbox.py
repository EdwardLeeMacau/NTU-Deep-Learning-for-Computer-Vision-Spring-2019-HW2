"""
  Filename     [ visualize_bbox.py ]
  PackageName  [ DLCV Spring 2019 - YOLOv1 ]
  Synposis     [  ]
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

DATA_CLASSES = (  # always index 0
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'small-vehicle', 'large-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field',
    'swimming-pool', 'container-crane')
  
Color = [
    [0, 0, 0],    [128, 0, 0],   [0, 128, 0],    [128, 128, 0],
    [0, 0, 128],  [128, 0, 128], [0, 128, 128],  [128, 128, 128],
    [64, 0, 0],   [192, 0, 0],   [64, 128, 0],   [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0],   [128, 64, 0],  [0, 192, 0],    [128, 192, 0],
    [0, 64, 128]
]
        
def parse_det(detfile):
    result = []

    with open(detfile, 'r') as f:
        for line in f:
            token = line.strip().split()
            
            # Ignore if the token has more than 10 elements
            if len(token) != 10:    
                continue
            
            x1 = int(float(token[0]))
            y1 = int(float(token[1]))
            x2 = int(float(token[4]))
            y2 = int(float(token[5]))
            className = token[8]
            prob = float(token[9])
            
            result.append([(x1,y1), (x2,y2), className, prob])
    
    return result 

def visualize(imgfile, detfile, outputfile):
    """
    Draw the bbox on 1 image

    Parameters
    ----------
    imgfile : numpy.array

    detfile : str
        The filename of detfile
    
    outputfile : str
        The filename of outputfile
    """
    image = cv2.imread(imgfile)
    result = parse_det(detfile)

    for left_up, right_bottom, class_name, prob in result:
        color = Color[DATA_CLASSES.index(class_name)]

        # Write the rectangle
        cv2.rectangle(image,left_up,right_bottom,color,2)
        
        # Write the labelName
        label = class_name + str(round(prob, 2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1] - text_size[1])
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

    cv2.imwrite(outputfile, image)

    return

def scan_folder(img_folder, det_folder, out_folder, size):
    """ 
    Scan the folder and make all photo with grids. 
    
    Parameters
    ----------
    img_folder, det_folder, out_folder : str
        (...)

    size : int
        The size of dataset
    """
    
    for i, name in enumerate(os.listdir(img_folder), 1):
        if (i % 100) == 0:  
            print(i)
        index = name.split(".")[0]

        imgfile = os.path.join(img_folder, index+".jpg")
        detfile = os.path.join(det_folder, index+".txt")

        visualize(imgfile, detfile, os.path.join(out_folder, index+".jpg"))

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./hw2_train_val/val1500", help="The root folder of the img/det. ")
    parser.add_argument("--number", type=int, default=1500)
    
    subparsers = parser.add_subparsers(required=True, dest="command")
    compare_parser = subparsers.add_parser("compare")
    drawdet_parser = subparsers.add_parser("drawdet")

    args = parser.parse_args()

    if args.command == "drawdet":
        """ Scan the whole folder. """
        imgfile = os.path.join(args.root, "images")
        detfile = os.path.join(args.root, "labelTxt_hbb_pred_sh")
        outfile = os.path.join(args.root, "images_pred")
        
        if not os.path.exists(outfile):
            os.mkdir(outfile)

        scan_folder(imgfile, detfile, outfile, args.number)

    elif args.command == "compare":
        """ Code from Ugo Tan """
        outputfolder = os.path.join(args.root, "images_pred")

        detfolder = os.path.join(args.root, "labelTxt_hbb_pred")
        txtnames = [ x.split(".")[0] for x in sorted(os.listdir(detfolder)) ]
        
        for index, img_det_name in enumerate(txtnames, 1):
            if index % 100 == 0: print(index)
            
            imgfile = os.path.join(args.root, "images", img_det_name+'.jpg')
            detfile = os.path.join(args.root, "labelTxt_hbb_pred", img_det_name+'.txt')
            GTfile  = os.path.join(args.root, "labelTxt_hbb", img_det_name+'.txt')

            image = cv2.imread(imgfile)
            gt_image = cv2.imread(imgfile)
            
            result = parse_det(detfile) # [(x1, y1), (x2, y2), cls, prob]
            for left_up, right_bottom, class_name, prob in result:
                color = Color[DATA_CLASSES.index(class_name)]
                cv2.rectangle(image,left_up,right_bottom,color,2)
                label = class_name + str(round(prob,2))
                text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                p1 = (left_up[0], left_up[1]- text_size[1])
                cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
                cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

            gt_img = parse_det(GTfile)
            for left_up, right_bottom, class_name, prob in gt_img:
                color = Color[DATA_CLASSES.index(class_name)]
                cv2.rectangle(gt_image,left_up,right_bottom,color,2)
                label = class_name+str(round(prob,2))
                text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                p1 = (left_up[0], left_up[1]- text_size[1])
                cv2.rectangle(gt_image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
                cv2.putText(gt_image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)
            
            pad = np.zeros((512, 20, 3))
            pad[:,:,:] = 255
            pad = pad.astype(np.uint8)

            final_pad = cv2.hconcat((image, pad))
            final = cv2.hconcat((final_pad, gt_image))

            cv2.imwrite(os.path.join(outputfolder, img_det_name+'_merge.jpg'), image)

if __name__ == '__main__':
    main()
