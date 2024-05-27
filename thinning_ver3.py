import cv2
import numpy as np
import os
from PIL import Image
import time
import math

import logging
from tqdm import tqdm

#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)

data_path = "../Data/goodResult_crop/"
binary_path = "binary_method/"
output_path = "output_method3/"
image_paths = os.listdir(data_path)

backgroundPixel = 0
objectPixel = 85
contourPixel = 170
skeletonPixel = 255

start = time.time()
math.factorial(100000)

for image in tqdm(image_paths):
    logging.debug(image)

    path = data_path + image
    
    floorplan = cv2.imread(path)    # image load

    height, width, _ = floorplan.shape

    floorplan_gray = cv2.cvtColor(floorplan, cv2.COLOR_BGR2GRAY)    # convert to gray-scale image

    _, floorplan_bin = cv2.threshold(floorplan_gray, 30, 255, cv2.THRESH_BINARY)    # thresholding (convert to binary image)

    #cv2.imwrite(output_path + image + "_bin.png", floorplan_bin)    # save binary image

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # generate kernel

    floorplan_morp = cv2.morphologyEx(floorplan_bin, cv2.MORPH_DILATE, kernel, iterations=3)    # morphology operation

    cv2.imwrite(output_path + image + "_bin.png", floorplan_morp)  # save morphology operated image

    lb = np.copy(floorplan_morp)    
    fp = np.copy(floorplan_morp)

    lb = cv2.bitwise_not(lb)

    y_indices, x_indices = np.where(lb == 255)
    for y, x in zip(y_indices, x_indices):
        lb[y, x] = objectPixel

    # cv2.imwrite(output_path + image + "_lb.png", lb)

    cnt = 1
    num = 1

    while cnt != 0:
        cnt = 0
        '''Preprocessing Stage'''
        logging.debug("Pre-processing(1) Start")
        y_indices1, x_indices1 = np.where(lb == objectPixel)
        for y, x in zip(y_indices1, x_indices1):
            if y == 0 or y == height-1 or x == 0 or x == width-1 :
                continue
            elif (lb[y-1, x] == backgroundPixel) or (lb[y, x-1] == backgroundPixel) or (lb[y, x+1] == backgroundPixel) or (lb[y+1, x] == backgroundPixel):
                lb[y, x] = contourPixel   # contour pixel
        logging.debug("Pre-processing(1) End")

        logging.debug("Pre-processing(1) Image Save")
        # cv2.imwrite(output_path + image + "_A1-" + str(num) + ".png", fp)
        # cv2.imwrite(output_path + image + "_A1_label-" + str(num) + ".png", lb)

        logging.debug("Pre-processing(2) Start")
        y_indices2, x_indices2 = np.where(lb == contourPixel)
        for y, x in zip(y_indices2, x_indices2):
            if y == 0 or y == height-1 or x == 0 or x == width-1 :
                continue
            elif (((lb[y, x+1] == backgroundPixel) and (lb[y+1, x+1] == backgroundPixel) and (lb[y+1, x] == backgroundPixel) and (lb[y+1, x-1] == backgroundPixel) and (lb[y, x-1] == backgroundPixel)) 
            and ((lb[y-1, x-1] != backgroundPixel) or (lb[y-1, x+1] != backgroundPixel))
            and (lb[y-1, x] != backgroundPixel)):
                fp[y, x] = 255
                lb[y, x] = backgroundPixel
                lb[y-1, x] = contourPixel
            elif (((lb[y-1, x-1] == backgroundPixel) and (lb[y-1, x] == backgroundPixel) and (lb[y, x-1] == backgroundPixel) and (lb[y+1, x-1] == backgroundPixel) and (lb[y+1, x] == backgroundPixel))
            and ((lb[y-1, x+1] != backgroundPixel) or (lb[y+1, x+1] != backgroundPixel))
            and (lb[y, x+1] != backgroundPixel)):
                fp[y, x] = 255
                lb[y, x] = backgroundPixel
                lb[y, x+1] = contourPixel
            elif (((lb[y-1, x-1] == backgroundPixel) and (lb[y-1, x] == backgroundPixel) and (lb[y-1, x+1] == backgroundPixel) and (lb[y, x-1] == backgroundPixel) and (lb[y, x+1] == backgroundPixel))
            and ((lb[y+1, x-1] != backgroundPixel) or (lb[y+1, x+1] != backgroundPixel))
            and (lb[y+1, x] != backgroundPixel)):
                fp[y, x] = 255
                lb[y, x] = backgroundPixel
                lb[y+1, x] = contourPixel
            elif (((lb[y-1, x] == backgroundPixel) and (lb[y-1, x+1] == backgroundPixel) and (lb[y, x+1] == backgroundPixel) and (lb[y+1, x] == backgroundPixel) and (lb[y+1, x+1] == backgroundPixel))
            and ((lb[y-1, x-1] != backgroundPixel) or (lb[y+1, x-1] != backgroundPixel))
            and (lb[y, x-1] != backgroundPixel)):
                fp[y, x] = 255
                lb[y, x] = backgroundPixel
                lb[y, x-1] = contourPixel

                '''Pre-processing Extension'''
            elif ((lb[y-1, x-1] == backgroundPixel) and (lb[y-1, x] == backgroundPixel) and (lb[y, x-1] == backgroundPixel) 
            and (lb[y-1, x+1] != backgroundPixel) and (lb[y, x+1] != backgroundPixel) and (lb[y+1, x-1] != backgroundPixel) and (lb[y+1, x] != backgroundPixel) and (lb[y+1, x+1] != backgroundPixel)):
                fp[y, x] = 255
                lb[y, x] = backgroundPixel
                lb[y, x+1] = contourPixel
                lb[y+1, x] = contourPixel
            elif ((lb[y-1, x] == backgroundPixel) and (lb[y-1, x+1] == backgroundPixel) and (lb[y, x+1] == backgroundPixel) 
            and (lb[y-1, x-1] != backgroundPixel) and (lb[y, x-1] != backgroundPixel) and (lb[y+1, x-1] != backgroundPixel) and (lb[y+1, x] != backgroundPixel) and (lb[y+1, x+1] != backgroundPixel)):
                fp[y, x] = 255
                lb[y, x] = backgroundPixel
                lb[y, x-1] = contourPixel
                lb[y+1, x] = contourPixel
            elif ((lb[y, x-1] == backgroundPixel) and (lb[y+1, x-1] == backgroundPixel) and (lb[y+1, x] == backgroundPixel) 
            and (lb[y-1, x-1] != backgroundPixel) and (lb[y-1, x] != backgroundPixel) and (lb[y-1, x+1] != backgroundPixel) and (lb[y, x+1] != backgroundPixel) and (lb[y+1, x+1] != backgroundPixel)):
                fp[y, x] = 255
                lb[y, x] = backgroundPixel
                lb[y-1, x] = contourPixel
                lb[y, x+1] = contourPixel
            elif ((lb[y, x+1] == backgroundPixel) and (lb[y+1, x] == backgroundPixel) and (lb[y+1, x+1] == backgroundPixel) 
            and (lb[y-1, x-1] != backgroundPixel) and (lb[y-1, x] != backgroundPixel) and (lb[y-1, x+1] != backgroundPixel) and (lb[y, x-1] != backgroundPixel) and (lb[y+1, x-1] != backgroundPixel)):
                fp[y, x] = 255
                lb[y, x] = backgroundPixel
                lb[y-1, x] = contourPixel
                lb[y, x-1] = contourPixel
        logging.debug("Pre-processing(2) End")
        
        logging.debug("Pre-processing(2) Image Save")
        # cv2.imwrite(output_path + image + "_A2-" + str(num) + ".png", fp)
        # cv2.imwrite(output_path + image + "_A2_label-" + str(num) + ".png", lb)


    
        '''Peeling Stage'''

        '''Keeping strokes one-pixel width'''
        lb_cp = np.copy(lb)
        
        logging.debug("Keeping strokes one-pixel width Start")
        y_indices3, x_indices3 = np.where(lb == contourPixel)
        for y, x in zip(y_indices3, x_indices3):
            if y == 0 or y == height-1 or x == 0 or x == width-1 :
                continue
            elif ((lb_cp[y-1, x] == backgroundPixel) and (lb_cp[y+1, x] == backgroundPixel)):
                lb[y, x] = skeletonPixel   # skeleton pixel
            elif ((lb_cp[y, x+1] == backgroundPixel) and (lb_cp[y, x-1] == backgroundPixel)):
                lb[y, x] = skeletonPixel
            elif (lb_cp[y-1, x-1] != backgroundPixel) and ((lb_cp[y-1, x] == backgroundPixel) and (lb_cp[y, x-1] == backgroundPixel)):
                lb[y, x] = skeletonPixel
            elif (lb_cp[y-1, x+1] != backgroundPixel) and ((lb_cp[y-1, x] == backgroundPixel) and (lb_cp[y, x+1] == backgroundPixel)):
                lb[y, x] = skeletonPixel
            elif (lb_cp[y+1, x+1] != backgroundPixel) and ((lb_cp[y, x+1] == backgroundPixel) and (lb_cp[y+1, x] == backgroundPixel)):
                lb[y, x] = skeletonPixel
            elif (lb_cp[y+1, x-1] != backgroundPixel) and ((lb_cp[y+1, x] == backgroundPixel) and (lb_cp[y, x-1] == backgroundPixel)):
                lb[y, x] = skeletonPixel
        logging.debug("Keeping strokes one-pixel width End")

        logging.debug("Keeping strokes one-pixel width Image Save")
        # cv2.imwrite(output_path + image + "_B-" + str(num) + ".png", fp)
        # cv2.imwrite(output_path + image + "_B_label-" + str(num) + ".png", lb)

        '''Removing pixels of corners'''
        # lb_cp = np.copy(lb)

        logging.debug("Removing pixels of corners Start")
        y_indices4, x_indices4 = np.where(lb == contourPixel)
        for y, x in zip(y_indices4, x_indices4):
            if y == 0 or y == height-1 or x == 0 or x == width-1 :
                continue
            elif (((lb_cp[y-1, x-1] == backgroundPixel) and (lb_cp[y-1, x] == backgroundPixel) and (lb_cp[y, x-1] == backgroundPixel)) 
            and (lb_cp[y, x+1] == contourPixel) 
            and (lb_cp[y+1, x] == contourPixel) 
            and (lb_cp[y+1, x+1] == objectPixel)): 
                fp[y, x] = 255
                lb[y, x] = backgroundPixel
                cnt += 1
            elif (((lb_cp[y-1, x] == backgroundPixel) and (lb_cp[y-1, x+1] == backgroundPixel) and (lb_cp[y, x+1] == backgroundPixel)) 
            and (lb_cp[y+1, x] == contourPixel) 
            and (lb_cp[y, x-1] == contourPixel) 
            and (lb_cp[y+1, x-1] == objectPixel)): 
                fp[y, x] = 255
                lb[y, x] = backgroundPixel
                cnt += 1
            elif (((lb_cp[y, x+1] == backgroundPixel) and (lb_cp[y+1, x+1] == backgroundPixel) and (lb_cp[y+1, x] == backgroundPixel)) 
            and (lb_cp[y-1, x] == contourPixel) 
            and (lb_cp[y, x-1] == contourPixel) 
            and (lb_cp[y-1, x-1] == objectPixel)): 
                fp[y, x] = 255
                lb[y, x] = backgroundPixel
                cnt += 1
            elif (((lb_cp[y+1, x] == backgroundPixel) and (lb_cp[y+1, x-1] == backgroundPixel) and (lb_cp[y, x-1] == backgroundPixel)) 
            and (lb_cp[y-1, x] == contourPixel) 
            and (lb_cp[y, x+1] == contourPixel) 
            and (lb_cp[y-1, x+1] == objectPixel)): 
                fp[y, x] = 255
                lb[y, x] = backgroundPixel
                cnt += 1
            elif (((lb_cp[y-1, x] == objectPixel) and (lb_cp[y+1, x] == backgroundPixel)) 
            or ((lb_cp[y-1, x] == backgroundPixel) and (lb_cp[y+1, x] == objectPixel))):
                fp[y, x] = 255
                lb[y, x] = backgroundPixel
                cnt += 1
            elif (((lb_cp[y, x-1] == objectPixel) and (lb_cp[y, x+1] == backgroundPixel)) 
            or ((lb_cp[y, x-1] == backgroundPixel) and (lb_cp[y, x+1] == objectPixel))):
                fp[y, x] = 255
                lb[y, x] = backgroundPixel
                cnt += 1
        logging.debug("Removing pixels of corners End")

        logging.debug("Removing pixels of corners Image Save")
        # cv2.imwrite(output_path + image + "_C-" + str(num) + ".png", fp)
        # cv2.imwrite(output_path + image + "_C_label-" + str(num) + ".png", lb)

        num += 1



    cnt = 1
    num = 1

    while cnt != 0:
        cnt = 0
        '''Remaining two-pixel width skeletonization'''

        logging.debug("Remaining two-pixel width skeletonization Start")
        y_indices5, x_indices5 = np.where((lb == contourPixel) | (lb == objectPixel))
        for y, x in zip(y_indices5, x_indices5):
            if y == 0 or y == height-1 or x == 0 or x == width-1 :
                continue
            #* left connectivity
            elif (lb[y, x-1] != backgroundPixel) and (lb[y+1, x-1] == backgroundPixel) and (lb[y+1, x] != backgroundPixel) and (lb[y+1, x+1] != backgroundPixel) and (lb[y, x+1] != backgroundPixel):
                if lb[y+2, x] == backgroundPixel:
                    fp[y+1, x] = 255
                    lb[y+1, x] = backgroundPixel
                    lb[y, x] = skeletonPixel
                    if lb[y-1, x] != backgroundPixel:
                        lb[y, x] = contourPixel
                    cnt += 1
                elif lb[y+2, x] != backgroundPixel:
                    fp[y+1, x+1] = 255
                    lb[y+1, x+1] = backgroundPixel
                    lb[y+1, x] = skeletonPixel
                    lb[y, x+1] = skeletonPixel
                    cnt += 1
            elif (lb[y, x-1] != backgroundPixel) and (lb[y-1, x-1] == backgroundPixel) and (lb[y-1, x] != backgroundPixel) and (lb[y-1, x+1] != backgroundPixel) and (lb[y, x+1] != backgroundPixel):
                if lb[y-2, x] == backgroundPixel:
                    fp[y-1, x] = 255
                    lb[y-1, x] = backgroundPixel
                    lb[y, x] = skeletonPixel
                    if lb[y+1, x] != backgroundPixel:
                        lb[y, x] = contourPixel
                    cnt += 1
                elif lb[y-2, x] != backgroundPixel:
                    fp[y-1, x+1] = 255
                    lb[y-1, x+1] = backgroundPixel
                    lb[y-1, x] = skeletonPixel
                    lb[y, x+1] = skeletonPixel
                    cnt += 1
            #* right connectivity
            elif (lb[y, x+1] != backgroundPixel) and (lb[y+1, x+1] == backgroundPixel) and (lb[y+1, x] != backgroundPixel) and (lb[y+1, x-1] != backgroundPixel) and (lb[y, x-1] != backgroundPixel):
                if lb[y+2, x] == backgroundPixel:
                    fp[y+1, x] = 255
                    lb[y+1, x] = backgroundPixel
                    lb[y, x] = skeletonPixel
                    if lb[y-1, x] != backgroundPixel:
                        lb[y, x] = contourPixel
                    cnt += 1
                elif lb[y+2, x] != backgroundPixel:
                    fp[y+1, x-1] = 255
                    lb[y+1, x-1] = backgroundPixel
                    lb[y+1, x] = skeletonPixel
                    lb[y, x-1] = skeletonPixel
                    cnt += 1
            elif (lb[y, x+1] != backgroundPixel) and (lb[y-1, x+1] == backgroundPixel) and (lb[y-1, x] != backgroundPixel) and (lb[y-1, x-1] != backgroundPixel) and (lb[y, x-1] != backgroundPixel):
                if lb[y-2, x] == backgroundPixel:
                    fp[y-1, x] = 255
                    lb[y-1, x] = backgroundPixel
                    lb[y, x] = skeletonPixel
                    if lb[y+1, x] != backgroundPixel:
                        lb[y, x] = contourPixel
                    cnt += 1
                elif lb[y-2, x] != backgroundPixel:
                    fp[y-1, x-1] = 255
                    lb[y-1, x-1] = backgroundPixel
                    lb[y-1, x] = skeletonPixel
                    lb[y, x-1] = skeletonPixel
                    cnt += 1
            #* up connectivity
            elif (lb[y-1, x] != backgroundPixel) and (lb[y-1, x+1] == backgroundPixel) and (lb[y, x+1] != backgroundPixel) and (lb[y+1, x+1] != backgroundPixel) and (lb[y+1, x] != backgroundPixel):
                if lb[y, x+2] == backgroundPixel:
                    fp[y, x+1] = 255
                    lb[y, x+1] = backgroundPixel
                    lb[y, x] = skeletonPixel
                    if lb[y, x-1] != backgroundPixel:
                        lb[y, x] = contourPixel
                    cnt += 1
                elif lb[y, x+2] != backgroundPixel:
                    fp[y+1, x+1] = 255
                    lb[y+1, x+1] = backgroundPixel
                    lb[y+1, x] = skeletonPixel
                    lb[y, x+1] = skeletonPixel
                    cnt += 1
            elif (lb[y-1, x] != backgroundPixel) and (lb[y-1, x-1] == backgroundPixel) and (lb[y, x-1] != backgroundPixel) and (lb[y+1, x-1] != backgroundPixel) and (lb[y+1, x] != backgroundPixel):
                if lb[y, x-2] == backgroundPixel:
                    fp[y, x-1] = 255
                    lb[y, x-1] = backgroundPixel
                    lb[y, x] = skeletonPixel
                    if lb[y, x+1] != backgroundPixel:
                        lb[y, x] = contourPixel
                    cnt += 1
                elif lb[y, x-2] != backgroundPixel:
                    fp[y+1, x-1] = 255
                    lb[y+1, x-1] = backgroundPixel
                    lb[y+1, x] = skeletonPixel
                    lb[y, x-1] = skeletonPixel
                    cnt += 1
            #* down connectivity
            elif (lb[y+1, x] != backgroundPixel) and (lb[y+1, x+1] == backgroundPixel) and (lb[y, x+1] != backgroundPixel) and (lb[y-1, x+1] != backgroundPixel) and (lb[y-1, x] != backgroundPixel):
                if lb[y, x+2] == backgroundPixel:
                    fp[y, x+1] = 255
                    lb[y, x+1] = backgroundPixel
                    lb[y, x] = skeletonPixel
                    if lb[y, x-1] != backgroundPixel:
                        lb[y, x] = contourPixel
                    cnt += 1
                elif lb[y, x+2] != backgroundPixel:
                    fp[y-1, x+1] = 255
                    lb[y-1, x+1] = backgroundPixel
                    lb[y-1, x] = skeletonPixel
                    lb[y, x+1] = skeletonPixel
                    cnt += 1
            elif (lb[y+1, x] != backgroundPixel) and (lb[y+1, x-1] == backgroundPixel) and (lb[y, x-1] != backgroundPixel) and (lb[y-1, x-1] != backgroundPixel) and (lb[y-1, x] != backgroundPixel):
                if lb[y, x-2] == backgroundPixel:
                    fp[y, x-1] = 255
                    lb[y, x-1] = backgroundPixel
                    lb[y, x] = skeletonPixel
                    if lb[y, x+1] != backgroundPixel:
                        lb[y, x] = contourPixel
                    cnt += 1
                elif lb[y, x-2] != backgroundPixel:
                    fp[y-1, x-1] = 255
                    lb[y-1, x-1] = backgroundPixel
                    lb[y-1, x] = skeletonPixel
                    lb[y, x-1] = skeletonPixel
                    cnt += 1
        
        logging.debug("Remaining two-pixel width skeletonization End")

        # cv2.imwrite(output_path + image + "_D-" + str(num) + ".png", fp)
        # cv2.imwrite(output_path + image + "_D_label-" + str(num) + ".png", lb)
        
        num += 1
    
    y_indices8, x_indices8 = np.where(lb != backgroundPixel)
    for y, x in zip(y_indices2, x_indices2):
        if y == 0 or y == height-1 or x == 0 or x == width-1 :
                continue
        elif (((lb[y, x+1] == backgroundPixel) and (lb[y+1, x+1] == backgroundPixel) and (lb[y+1, x] == backgroundPixel) and (lb[y+1, x-1] == backgroundPixel) and (lb[y, x-1] == backgroundPixel)) 
        and ((lb[y-1, x-1] != backgroundPixel) or (lb[y-1, x+1] != backgroundPixel))
        and (lb[y-1, x] != backgroundPixel)):
            fp[y, x] = 255
            lb[y, x] = backgroundPixel
            lb[y-1, x] = contourPixel
        elif (((lb[y-1, x-1] == backgroundPixel) and (lb[y-1, x] == backgroundPixel) and (lb[y, x-1] == backgroundPixel) and (lb[y+1, x-1] == backgroundPixel) and (lb[y+1, x] == backgroundPixel))
        and ((lb[y-1, x+1] != backgroundPixel) or (lb[y+1, x+1] != backgroundPixel))
        and (lb[y, x+1] != backgroundPixel)):
            fp[y, x] = 255
            lb[y, x] = backgroundPixel
            lb[y, x+1] = contourPixel
        elif (((lb[y-1, x-1] == backgroundPixel) and (lb[y-1, x] == backgroundPixel) and (lb[y-1, x+1] == backgroundPixel) and (lb[y, x-1] == backgroundPixel) and (lb[y, x+1] == backgroundPixel))
        and ((lb[y+1, x-1] != backgroundPixel) or (lb[y+1, x+1] != backgroundPixel))
        and (lb[y+1, x] != backgroundPixel)):
            fp[y, x] = 255
            lb[y, x] = backgroundPixel
            lb[y+1, x] = contourPixel
        elif (((lb[y-1, x] == backgroundPixel) and (lb[y-1, x+1] == backgroundPixel) and (lb[y, x+1] == backgroundPixel) and (lb[y+1, x] == backgroundPixel) and (lb[y+1, x+1] == backgroundPixel))
        and ((lb[y-1, x-1] != backgroundPixel) or (lb[y+1, x-1] != backgroundPixel))
        and (lb[y, x-1] != backgroundPixel)):
            fp[y, x] = 255
            lb[y, x] = backgroundPixel
            lb[y, x-1] = contourPixel
            
        # elif ((lb[y-1, x-1] == backgroundPixel) and (lb[y-1, x] == backgroundPixel) and (lb[y, x-1] == backgroundPixel) 
        # and (lb[y-1, x+1] != backgroundPixel) and (lb[y, x+1] != backgroundPixel) and (lb[y+1, x-1] != backgroundPixel) and (lb[y+1, x] != backgroundPixel)):
        #     fp[y, x] = 255
        #     lb[y, x] = backgroundPixel
        #     lb[y, x+1] = contourPixel
        #     lb[y+1, x] = contourPixel
        # elif ((lb[y-1, x] == backgroundPixel) and (lb[y-1, x+1] == backgroundPixel) and (lb[y, x+1] == backgroundPixel) 
        # and (lb[y-1, x-1] != backgroundPixel) and (lb[y, x-1] != backgroundPixel)  and (lb[y+1, x] != backgroundPixel) and (lb[y+1, x+1] != backgroundPixel)):
        #     fp[y, x] = 255
        #     lb[y, x] = backgroundPixel
        #     lb[y, x-1] = contourPixel
        #     lb[y+1, x] = contourPixel
        # elif ((lb[y, x-1] == backgroundPixel) and (lb[y+1, x-1] == backgroundPixel) and (lb[y+1, x] == backgroundPixel) 
        # and (lb[y-1, x-1] != backgroundPixel) and (lb[y-1, x] != backgroundPixel) and (lb[y, x+1] != backgroundPixel) and (lb[y+1, x+1] != backgroundPixel)):
        #     fp[y, x] = 255
        #     lb[y, x] = backgroundPixel
        #     lb[y-1, x] = contourPixel
        #     lb[y, x+1] = contourPixel
        # elif ((lb[y, x+1] == backgroundPixel) and (lb[y+1, x] == backgroundPixel) and (lb[y+1, x+1] == backgroundPixel) 
        # and (lb[y-1, x] != backgroundPixel) and (lb[y-1, x+1] != backgroundPixel) and (lb[y, x-1] != backgroundPixel) and (lb[y+1, x-1] != backgroundPixel)):
        #     fp[y, x] = 255
        #     lb[y, x] = backgroundPixel
        #     lb[y-1, x] = contourPixel
        #     lb[y, x-1] = contourPixel

    '''Post-processing Stage'''

    '''Two-pixel width solving'''
    cnt = 1
    num = 1
    while cnt != 0:
        cnt = 0
        y_indices6, x_indices6 = np.where(lb == contourPixel)
        for y, x in zip(y_indices6, x_indices6):
            if y == 0 or y == height-1 or x == 0 or x == width-1 :
                continue
            elif ((lb[y-1, x-1] == backgroundPixel) and (lb[y-1, x] == backgroundPixel) and (lb[y, x-1] == backgroundPixel) and (lb[y+1, x+1] == objectPixel)):
                fp[y, x] = 255
                lb[y, x] = backgroundPixel
                cnt += 1
            elif ((lb[y-1, x] == backgroundPixel) and (lb[y-1, x+1] == backgroundPixel) and (lb[y, x+1] == backgroundPixel) and (lb[y+1, x-1] == objectPixel)):
                fp[y, x] = 255
                lb[y, x] = backgroundPixel
                cnt += 1
            elif ((lb[y, x+1] == backgroundPixel) and (lb[y+1, x+1] == backgroundPixel) and (lb[y+1, x] == backgroundPixel) and (lb[y-1, x-1] == objectPixel)):
                fp[y, x] = 255
                lb[y, x] = backgroundPixel
                cnt += 1
            elif ((lb[y+1, x] == backgroundPixel) and (lb[y+1, x-1] == backgroundPixel) and (lb[y, x-1] == backgroundPixel) and (lb[y-1, x+1] == objectPixel)):
                fp[y, x] = 255
                lb[y, x] = backgroundPixel
                cnt += 1
        # cv2.imwrite(output_path + image + "_E-" + str(num) + ".png", fp)
        # cv2.imwrite(output_path + image + "_E_label-" + str(num) + ".png", lb)

        num += 1

    '''Stairs solving'''
    y_indices7, x_indices7 = np.where(lb == contourPixel)
    for y, x in zip(y_indices7, x_indices7):
        if y == 0 or y == height-1 or x == 0 or x == width-1 :
                continue
        elif (lb[y-1, x] != backgroundPixel) and (lb[y, x-1] != backgroundPixel) and (lb[y-1, x-1] == backgroundPixel):
            fp[y, x] = 255
            lb[y, x] = backgroundPixel
        elif (lb[y-1, x] != backgroundPixel) and (lb[y, x+1] != backgroundPixel) and (lb[y-1, x+1] == backgroundPixel):
            fp[y, x] = 255
            lb[y, x] = backgroundPixel
        elif (lb[y, x+1] != backgroundPixel) and (lb[y+1, x] != backgroundPixel) and (lb[y+1, x+1] == backgroundPixel):
            fp[y, x] = 255
            lb[y, x] = backgroundPixel
        elif (lb[y+1, x] != backgroundPixel) and (lb[y, x-1] != backgroundPixel) and (lb[y+1, x-1] == backgroundPixel):
            fp[y, x] = 255
            lb[y, x] = backgroundPixel

    cv2.imwrite(output_path + image + "_result.png", fp)
    # cv2.imwrite(output_path + image + "_result_label.png", lb)

end = time.time()
logging.info(f"{end - start: .5f} sec")