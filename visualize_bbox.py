#encoding:utf-8
#
#created by xiongzihua
#

import sys
import os

import cv2
import numpy as np

DOTA_CLASSES = (  # always index 0
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'small-vehicle', 'large-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field',
    'swimming-pool', 'container-crane')

Color = [[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]]
        
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

def scan_folder(img_folder, det_folder, out_folder, size):
    """ scan the folder and make all photo with grids. """
    length = len(str(size))

    for i in range(0, size):
        if (i % 100) == 0:  print(i)
        index = str(i).zfill(length)

        imgfile = os.path.join(img_folder, index+".jpg")
        detfile = os.path.join(det_folder, index+".txt")

        visualize(imgfile, detfile, os.path.join(out_folder, index+".jpg"))

def visualize(imgfile, detfile, outputfile):
    image = cv2.imread(imgfile)
    result = parse_det(detfile)

    for left_up, right_bottom, class_name, prob in result:
        color = Color[DOTA_CLASSES.index(class_name)]
        cv2.rectangle(image,left_up,right_bottom,color,2)
        label = class_name + str(round(prob,2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1]- text_size[1])
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

    cv2.imwrite(outputfile, image)

def main():
    """ Scan the whole folder. """
    rootpath = sys.argv[1]
    number   = int(sys.argv[2])
    imgfile = os.path.join(rootpath, "images")
    detfile = os.path.join(rootpath, "labelTxt_hbb_pred")
    outfile = os.path.join(rootpath, "images_pred")
    
    if not os.path.exists(outfile):
        os.mkdir(outfile)

    scan_folder(imgfile, detfile, outfile, number)

if __name__ == '__main__':
    main()
