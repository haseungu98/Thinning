import cv2
import numpy as np
import os
from PIL import Image
import time
import math

data_path = "../Data/goodResult_crop/"
output_path = "output_ver2/"
image_paths = os.listdir(data_path)

backgroundPixel = 0
objectPixel = 85
contourPixel = 170
skeletonPixel = 255

start = time.time()
math.factorial(100000)

for image in image_paths:
    print(image)

    path = data_path + image
    
    floorplan = cv2.imread(path)    # image load

    height, width, _ = floorplan.shape

    floorplan_gray = cv2.cvtColor(floorplan, cv2.COLOR_BGR2GRAY)    # convert to gray-scale image

    _, floorplan_bin = cv2.threshold(floorplan_gray, 80, 255, cv2.THRESH_BINARY)    # thresholding (convert to binary image)

    #cv2.imwrite(output_path + image + "_bin.png", floorplan_bin)    # save binary image

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # generate kernel

    floorplan_morp = cv2.morphologyEx(floorplan_bin, cv2.MORPH_DILATE, kernel, iterations=2)    # morphology operation

    cv2.imwrite(output_path + image + "_bin.png", floorplan_morp)  # save morphology operated image

    label = np.copy(floorplan_morp)
    floorplan_thin = np.copy(floorplan_morp)

    label = cv2.bitwise_not(label)

    y_indices, x_indices = np.where(label == 255)
    for y, x in zip(y_indices, x_indices):
        label[y, x] = objectPixel

    cv2.imwrite(output_path + image + "_label_bin.png", label)

    cnt = 1
    num = 1

    while cnt != 0:

        '''Pre-processing Stage'''
        print("Pre-processing(1) Start")
        y_indices1, x_indices1 = np.where(label == objectPixel)
        for y, x in zip(y_indices1, x_indices1):

            if (label[y-1, x] == backgroundPixel) or (label[y, x-1] == backgroundPixel) or (label[y, x+1] == backgroundPixel) or (label[y+1, x] == backgroundPixel):
                label[y, x] = contourPixel   # contour pixel
        print("Pre-processing(1) End")
        
        print("Pre-processing(1) Image Save")
        cv2.imwrite(output_path + image + "_pre(1)-" + str(num) + ".png", floorplan_thin)
        cv2.imwrite(output_path + image + "_label_pre(1)-" + str(num) + ".png", label)

        print("Pre-processing(2) Start")
        y_indices2, x_indices2 = np.where(label == contourPixel)
        for y, x in zip(y_indices2, x_indices2):

            if (((label[y, x+1] == backgroundPixel) and (label[y+1, x+1] == backgroundPixel) and (label[y+1, x] == backgroundPixel) and (label[y+1, x-1] == backgroundPixel) and (label[y, x-1] == backgroundPixel)) 
            and ((label[y-1, x-1] != backgroundPixel) or (label[y-1, x+1] != backgroundPixel))
            and (label[y-1, x] != backgroundPixel)):
                floorplan_thin[y, x] = 255
                label[y, x] = backgroundPixel
                label[y-1, x] = contourPixel

            elif (((label[y-1, x-1] == backgroundPixel) and (label[y-1, x] == backgroundPixel) and (label[y, x-1] == backgroundPixel) and (label[y+1, x-1] == backgroundPixel) and (label[y+1, x] == backgroundPixel))
            and ((label[y-1, x+1] != backgroundPixel) or (label[y+1, x+1] != backgroundPixel))
            and (label[y, x+1] != backgroundPixel)):
                floorplan_thin[y, x] = 255
                label[y, x] = backgroundPixel
                label[y, x+1] = contourPixel

            elif (((label[y-1, x-1] == backgroundPixel) and (label[y-1, x] == backgroundPixel) and (label[y-1, x+1] == backgroundPixel) and (label[y, x-1] == backgroundPixel) and (label[y, x+1] == backgroundPixel))
            and ((label[y+1, x-1] != backgroundPixel) or (label[y+1, x+1] != backgroundPixel))
            and (label[y+1, x] != backgroundPixel)):
                floorplan_thin[y, x] = 255
                label[y, x] = backgroundPixel
                label[y+1, x] = contourPixel

            elif (((label[y-1, x] == backgroundPixel) and (label[y-1, x+1] == backgroundPixel) and (label[y, x+1] == backgroundPixel) and (label[y+1, x] == backgroundPixel) and (label[y+1, x+1] == backgroundPixel))
            and ((label[y-1, x-1] != backgroundPixel) or (label[y+1, x-1] != backgroundPixel))
            and (label[y, x-1] != backgroundPixel)):
                floorplan_thin[y, x] = 255
                label[y, x] = backgroundPixel
                label[y, x-1] = contourPixel

                '''Pre-processing Expand'''
            elif ((label[y-1, x-1] == backgroundPixel) and (label[y-1, x] == backgroundPixel) and (label[y, x-1] == backgroundPixel) 
            and (label[y-1, x+1] != backgroundPixel) and (label[y, x+1] != backgroundPixel) and (label[y+1, x-1] != backgroundPixel) and (label[y+1, x] != backgroundPixel) and (label[y+1, x+1] != backgroundPixel)):
                floorplan_thin[y, x] = 255
                label[y, x] = backgroundPixel
                label[y, x+1] = contourPixel
                label[y+1, x] = contourPixel

            elif ((label[y-1, x] == backgroundPixel) and (label[y-1, x+1] == backgroundPixel) and (label[y, x+1] == backgroundPixel) 
            and (label[y-1, x-1] != backgroundPixel) and (label[y, x-1] != backgroundPixel) and (label[y+1, x-1] != backgroundPixel) and (label[y+1, x] != backgroundPixel) and (label[y+1, x+1] != backgroundPixel)):
                floorplan_thin[y, x] = 255
                label[y, x] = backgroundPixel
                label[y, x-1] = contourPixel
                label[y+1, x] = contourPixel

            elif ((label[y, x-1] == backgroundPixel) and (label[y+1, x-1] == backgroundPixel) and (label[y+1, x] == backgroundPixel) 
            and (label[y-1, x-1] != backgroundPixel) and (label[y-1, x] != backgroundPixel) and (label[y-1, x+1] != backgroundPixel) and (label[y, x+1] != backgroundPixel) and (label[y+1, x+1] != backgroundPixel)):
                floorplan_thin[y, x] = 255
                label[y, x] = backgroundPixel
                label[y-1, x] = contourPixel
                label[y, x+1] = contourPixel

            elif ((label[y, x+1] == backgroundPixel) and (label[y+1, x] == backgroundPixel) and (label[y+1, x+1] == backgroundPixel) 
            and (label[y-1, x-1] != backgroundPixel) and (label[y-1, x] != backgroundPixel) and (label[y-1, x+1] != backgroundPixel) and (label[y, x-1] != backgroundPixel) and (label[y+1, x-1] != backgroundPixel)):
                floorplan_thin[y, x] = 255
                label[y, x] = backgroundPixel
                label[y-1, x] = contourPixel
                label[y, x-1] = contourPixel
        print("Pre-processing(2) End")

        print("Pre-processing(2) Image Save")
        cv2.imwrite(output_path + image + "_pre(2)-" + str(num) + ".png", floorplan_thin)
        cv2.imwrite(output_path + image + "_label_pre(2)-" + str(num) + ".png", label)

        '''Peeling Stage'''
        cnt = 0

        label_copy = np.copy(label)

        '''Keeping strokes one-pixel width'''
        print("Keeping strokes one-pixel width Start")
        y_indices3, x_indices3 = np.where(label_copy == contourPixel)
        for y, x in zip(y_indices3, x_indices3):

            if ((label_copy[y-1, x] == backgroundPixel) and (label_copy[y+1, x] == backgroundPixel)):
                label[y, x] = skeletonPixel   # skeleton pixel

            elif ((label_copy[y, x+1] == backgroundPixel) and (label_copy[y, x-1] == backgroundPixel)):
                label[y, x] = skeletonPixel

            elif (label_copy[y-1, x-1] != backgroundPixel) and ((label_copy[y-1, x] == backgroundPixel) and (label_copy[y, x-1] == backgroundPixel)):
                label[y, x] = skeletonPixel

            elif (label_copy[y-1, x+1] != backgroundPixel) and ((label_copy[y-1, x] == backgroundPixel) and (label_copy[y, x+1] == backgroundPixel)):
                label[y, x] = skeletonPixel

            elif (label_copy[y+1, x+1] != backgroundPixel) and ((label_copy[y, x+1] == backgroundPixel) and (label_copy[y+1, x] == backgroundPixel)):
                label[y, x] = skeletonPixel

            elif (label_copy[y+1, x-1] != backgroundPixel) and ((label_copy[y+1, x] == backgroundPixel) and (label_copy[y, x-1] == backgroundPixel)):
                label[y, x] = skeletonPixel
        print("Keeping strokes one-pixel width End")

        print("Keeping strokes one-pixel width Image Save")
        cv2.imwrite(output_path + image + "_ksopw" + str(num) + ".png", floorplan_thin)
        cv2.imwrite(output_path + image + "_label_ksopw" + str(num) + ".png", label)

        '''Removing pixels of corners'''
        print("Removing pixels of corners Start")
        y_indices4, x_indices4 = np.where(label_copy == contourPixel)
        for y, x in zip(y_indices4, x_indices4):

            if (((label_copy[y-1, x-1] == backgroundPixel) and (label_copy[y-1, x] == backgroundPixel) and (label_copy[y, x-1] == backgroundPixel)) 
            and (label_copy[y, x+1] == contourPixel) 
            and (label_copy[y+1, x] == contourPixel) 
            and (label_copy[y+1, x+1] == objectPixel)): 
                floorplan_thin[y, x] = 255
                label[y, x] = backgroundPixel
                cnt += 1

            elif (((label_copy[y-1, x] == backgroundPixel) and (label_copy[y-1, x+1] == backgroundPixel) and (label_copy[y, x+1] == backgroundPixel)) 
            and (label_copy[y+1, x] == contourPixel) 
            and (label_copy[y, x-1] == contourPixel) 
            and (label_copy[y+1, x-1] == objectPixel)): 
                floorplan_thin[y, x] = 255
                label[y, x] = backgroundPixel
                cnt += 1

            elif (((label_copy[y, x+1] == backgroundPixel) and (label_copy[y+1, x+1] == backgroundPixel) and (label_copy[y+1, x] == backgroundPixel)) 
            and (label_copy[y-1, x] == contourPixel) 
            and (label_copy[y, x-1] == contourPixel) 
            and (label_copy[y-1, x-1] == objectPixel)): 
                floorplan_thin[y, x] = 255
                label[y, x] = backgroundPixel
                cnt += 1

            elif (((label_copy[y+1, x] == backgroundPixel) and (label_copy[y+1, x-1] == backgroundPixel) and (label_copy[y, x-1] == backgroundPixel)) 
            and (label_copy[y-1, x] == contourPixel) 
            and (label_copy[y, x+1] == contourPixel) 
            and (label_copy[y-1, x+1] == objectPixel)): 
                floorplan_thin[y, x] = 255
                label[y, x] = backgroundPixel
                cnt += 1

            elif (((label_copy[y-1, x] == objectPixel) and (label_copy[y+1, x] == backgroundPixel)) 
            or ((label_copy[y-1, x] == backgroundPixel) and (label_copy[y+1, x] == objectPixel))):
                floorplan_thin[y, x] = 255
                label[y, x] = backgroundPixel
                cnt += 1

            elif (((label_copy[y, x-1] == objectPixel) and (label_copy[y, x+1] == backgroundPixel)) 
            or ((label_copy[y, x-1] == backgroundPixel) and (label_copy[y, x+1] == objectPixel))):
                floorplan_thin[y, x] = 255
                label[y, x] = backgroundPixel
                cnt += 1
        print("Removing pixels of corners End")

        print("Removing pixels of corners Image Save")
        cv2.imwrite(output_path + image + "_rpc" + str(num) + ".png", floorplan_thin)
        cv2.imwrite(output_path + image + "_label_rpc" + str(num) + ".png", label)

        num += 1

        # print("Keeping strokes one-pixel width Start")
        # y_indices7, x_indices7 = np.where(label_copy == contourPixel)
        # for y, x in zip(y_indices7, x_indices7):

        #     if ((label[y-1, x] == skeletonPixel) or (label[y, x+1] == skeletonPixel) or (label[y+1, x] == skeletonPixel) or (label[y, x-1] == skeletonPixel)):
        #         label[y, x] = skeletonPixel



    '''Post-processing Stage'''

    '''Two-pixel width solving'''
    # cnt = 1
    # while cnt != 0:

    #     cnt = 0

    #     y_indices5, x_indices5 = np.where(label == objectPixel)
    #     for y, x in zip(y_indices5, x_indices5):

    #         if (((label[y-1, x-1] == backgroundPixel) and (label[y-1, x] == backgroundPixel) and (label[y, x-1] == backgroundPixel)) and (label[y+1, x+1] == objectPixel)):
    #             floorplan_thin[y, x] = 255
    #             label[y, x] = backgroundPixel
    #             cnt += 1

    #         elif (((label[y-1, x] == backgroundPixel) and (label[y-1, x+1] == backgroundPixel) and (label[y, x+1] == backgroundPixel)) and (label[y+1, x-1] == objectPixel)):
    #             floorplan_thin[y, x] = 255
    #             label[y, x] = backgroundPixel
    #             cnt += 1

    #         elif (((label[y, x+1] == backgroundPixel) and (label[y+1, x+1] == backgroundPixel) and (label[y+1, x] == backgroundPixel)) and (label[y-1, x-1] == objectPixel)):
    #             floorplan_thin[y, x] = 255
    #             label[y, x] = backgroundPixel
    #             cnt += 1

    #         elif (((label[y+1, x] == backgroundPixel) and (label[y+1, x-1] == backgroundPixel) and (label[y, x-1] == backgroundPixel)) and (label[y-1, x+1] == objectPixel)):
    #             floorplan_thin[y, x] = 255
    #             label[y, x] = backgroundPixel
    #             cnt += 1

    '''Stairs solving'''
    # y_indices6, x_indices6 = np.where(label == objectPixel)
    # for y, x in zip(y_indices6, x_indices6):

    #     if ((label[y-1, x] != backgroundPixel) and (label[y, x-1] != backgroundPixel)) and (label[y-1, x-1] == backgroundPixel):
    #         floorplan_thin[y, x] = 255
    #         label[y, x] = backgroundPixel

    #     elif ((label[y-1, x] != backgroundPixel) and (label[y, x+1] != backgroundPixel)) and (label[y-1, x+1] == backgroundPixel):
    #         floorplan_thin[y, x] = 255
    #         label[y, x] = backgroundPixel

    #     elif ((label[y, x+1] != backgroundPixel) and (label[y+1, x] != backgroundPixel)) and (label[y+1, x+1] == backgroundPixel):
    #         floorplan_thin[y, x] = 255
    #         label[y, x] = backgroundPixel

    #     elif ((label[y+1, x] != backgroundPixel) and (label[y, x-1] != backgroundPixel)) and (label[y+1, x-1] == backgroundPixel):
    #         floorplan_thin[y, x] = 255
    #         label[y, x] = backgroundPixel

    # cv2.imwrite(output_path + image + "_skeleton.png", floorplan_thin)
    # cv2.imwrite(output_path + image + "_label_skel.png", label)
                         
end = time.time()
print(f"{end - start: .5f} sec")