import cv2
import numpy as np
import os
from PIL import Image
import time
import math
import natsort

import logging
from tqdm import tqdm

data_path = "output_Zhang3/"
img_paths = natsort.natsorted(os.listdir(data_path))
img_num = len(img_paths)

ttco = [0] * (int(img_num / 2))
ttcs = [0] * (int(img_num / 2))

num = 0

tr_total = 0
acc = 0

for image in tqdm(img_paths):

    path = data_path + image
    
    # print(path)
    
    img = cv2.imread(path)    # image load
    
    height, width, _ = img.shape
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # convert to gray-scale image

    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    
    y_indices, x_indices = np.where(img == 0)
    for y, x in zip(y_indices, x_indices):
        if y == 0 or y == height-1 or x == 0 or x == width-1 :
                continue
        else:
            if img[y-1, x-1] == 0 and img[y-1, x] == 0:
                if "bin" in path:
                    ttco[int(math.floor(num/2))] += 1
                else:
                    ttcs[int(math.floor(num/2))] += 1
            if img[y-1, x] == 0 and img[y-1, x+1] == 0:
                if "bin" in path:
                    ttco[int(math.floor(num/2))] += 1
                else:
                    ttcs[int(math.floor(num/2))] += 1
            if img[y-1, x+1] == 0 and img[y, x+1] == 0:
                if "bin" in path:
                    ttco[int(math.floor(num/2))] += 1
                else:
                    ttcs[int(math.floor(num/2))] += 1
            if img[y, x+1] == 0 and img[y+1, x+1] == 0:
                if "bin" in path:
                    ttco[int(math.floor(num/2))] += 1
                else:
                    ttcs[int(math.floor(num/2))] += 1
            if img[y+1, x+1] == 0 and img[y+1, x] == 0:
                if "bin" in path:
                    ttco[int(math.floor(num/2))] += 1
                else:
                    ttcs[int(math.floor(num/2))] += 1
            if img[y+1, x] == 0 and img[y+1, x-1] == 0:
                if "bin" in path:
                    ttco[int(math.floor(num/2))] += 1
                else:
                    ttcs[int(math.floor(num/2))] += 1
            if img[y+1, x-1] == 0 and img[y, x-1] == 0:
                if "bin" in path:
                    ttco[int(math.floor(num/2))] += 1
                else:
                    ttcs[int(math.floor(num/2))] += 1
            if img[y, x-1] == 0 and img[y-1, x-1] == 0:
                if "bin" in path:
                    ttco[int(math.floor(num/2))] += 1
                else:
                    ttcs[int(math.floor(num/2))] += 1
        
    # print(ttco)
    # print(ttcs)
    
    if ttcs[int(math.floor(num/2))] == 0:
        ttcs[int(math.floor(num/2))] = 1
    
    if num % 2 == 0:
        tr = (1 - (ttcs[int(math.floor(num/2))] / ttco[int(math.floor(num/2))])) * 100
        tr_total += tr
    
    num += 1
    
acc = tr_total / int(img_num / 2)

print (acc)