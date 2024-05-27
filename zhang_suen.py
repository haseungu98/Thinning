import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import Counter

input_path = '../Data/goodResult_crop/'
binary_path = 'binary_Zhang/'
output_path = 'binary_Zhang/'
thinning_path = 'output_Zhang3/'


def binarization(file_name):
    img = cv2.imread(input_path + file_name, cv2.IMREAD_GRAYSCALE)

    _, binary_image = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # generate kernel

    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_DILATE, kernel, iterations=3)    # morphology operation

    cv2.imwrite(thinning_path + file_name + "_bin.png", binary_image)


def calculate_continuous_zeroes(image, inlier = 0.03, min_length=5):
    def calculate_in_direction(data, threshold, min_length):
        continuous_lengths = []
        for line in data:
            if not np.all(line == 255):  # Check if the line is not all 255s
                zero_indices = np.where(line == 0)[0]
                if len(zero_indices) > 0:
                    start = zero_indices[0]
                    length = 1

                    for index in zero_indices[1:]:
                        if index == start + length:  # Check if indices are continuous
                            length += 1
                        else:
                            if length >= threshold or length < min_length:
                                break  # Stop if length exceeds threshold
                            continuous_lengths.append(length)
                            start = index
                            length = 1

                    # Check the last continuous segment
                    if length < threshold and length >= min_length:
                        continuous_lengths.append(length)
        return continuous_lengths

    rows, cols = image.shape
    row_lengths = calculate_in_direction(image, inlier * cols, min_length)
    col_lengths = calculate_in_direction(image.T, inlier * rows, min_length)

    return row_lengths + col_lengths  # Combine row and column lengths

def get_two_largest_counts(lengths):
    """
    Return the two largest counts in the histogram.
    """
    counter = Counter(lengths)
    most_common = counter.most_common(2)  # Get two most common lengths

    return most_common

def plot_histogram(lengths, file_name=None):
    plt.figure()
    plt.hist(lengths, bins=range(1, max(lengths) + 2), align='left')
    plt.title('Histogram of Continuous Zero Lengths')
    plt.xlabel('Length of Continuous Zeroes')
    plt.ylabel('Frequency')
    
    if file_name is None:
        plt.show()
    else:
        plt.savefig(f'{output_path}histo_{file_name}')


def wall_extraction(img, kernel_size, kernel2_size = None):

    mask_map = cv2.bitwise_not(np.zeros_like(img)) / 255
    mask_map2 = cv2.bitwise_not(np.zeros_like(img)) / 255

    pad_size = kernel_size // 2
    pad_img = np.pad(img, pad_size, mode='constant', constant_values=0)

    if kernel2_size is None:
        k2_offset = (kernel_size) // 4
    else:
        k2_offset = kernel2_size // 2

    y_indices, x_indices = np.where(img == 0)
    for y, x in zip(y_indices, x_indices):
        kernel_region = pad_img[y:y+kernel_size,
                                 x:x+kernel_size]
        if np.all(kernel_region == 0):
            mask_map[y,x] = 0
        elif np.all(kernel_region[k2_offset:-k2_offset,k2_offset:-k2_offset] == 0):
            mask_map2[y,x] = 0
    return mask_map, mask_map2


def zhang_suen_thinning(img):
    # 이미지를 0과 1로 변환
    img = cv2.bitwise_not(img)//255

    def neighbors(y, x):
        """ 이미지에서 주어진 좌표의 8-이웃을 반환 """
        # 8-이웃 좌표
        n_coords = [(y-1, x), (y-1, x+1), (y, x+1), (y+1, x+1),
                    (y+1, x), (y+1, x-1), (y, x-1), (y-1, x-1)]
        return [img[ny, nx] if 0 <= nx < img.shape[1] and 0 <= ny < img.shape[0] else 0 for ny, nx in n_coords]

    def transitions(neigh):
        """ 0에서 1로 전환하는 횟수를 계산 """
        n = neigh + [neigh[0]]
        return sum((p1, p2) == (0, 1) for p1, p2 in zip(n, n[1:]))

    def thinning_iteration(img, iter):
        # 픽셀 제거 표시를 위한 마스크 생성
        marker = np.zeros(img.shape, dtype=bool)
        y_indices, x_indices = np.where(img)
        for y, x in zip(y_indices, x_indices):
            P = neighbors(y, x)
            A = transitions(P)
            B = sum(P)
            conditions = [
                2 <= B <= 6,
                A == 1,
                P[0] * P[2] * P[4] == 0 if iter == 0 else P[0] * P[2] * P[6] == 0,
                P[2] * P[4] * P[6] == 0 if iter == 0 else P[0] * P[4] * P[6] == 0
            ]
            # 모든 조건이 참이면 해당 픽셀을 제거
            if all(conditions):
                marker[y, x] = True
        return img & ~marker

    # 세선화 반복
    prev_img = np.zeros(img.shape, np.uint8)
    while not np.array_equal(img, prev_img):
        prev_img = img.copy()
        # 두 서브-이터레이션
        img = thinning_iteration(img, 0)
        img = thinning_iteration(img, 1)

    return cv2.bitwise_not(img * 255)


def wall_thining(file_name):
    img = cv2.imread(thinning_path + file_name + "_bin.png", cv2.IMREAD_GRAYSCALE)
    result = zhang_suen_thinning(img)
    cv2.imwrite(thinning_path + file_name + "_result.png", result)
    

def wall_seg_from_histo(file_name):
    # Load the binary image
    img = cv2.imread(binary_path + file_name, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

    # Calculate continuous zeroes
    continuous_zero_lengths = calculate_continuous_zeroes(img)

    two_largest_counts = get_two_largest_counts(continuous_zero_lengths)

    # print(f'kernel_size: ({two_largest_counts[0][0]}, {two_largest_counts[1][0]})')
    # Plot histogram
    plot_histogram(continuous_zero_lengths, file_name)


    mg, mg2 = wall_extraction(img, np.max([two_largest_counts[0][0], two_largest_counts[1][0]]))

    mg = mg * 255
    mg2 = mg2 * 255

    # cv2.imshow('mg', mg)
    # cv2.imshow('mg2', mg2)

    # mg = cv2.bitwise_not(mg)
    # mg2 = cv2.bitwise_not(mg2)

    # cv2.imshow('mgmg', mg)
    # cv2.imshow('mgmg2', mg2)

    # cv2.waitKey()

    # print(mg)
    # print(mg2)

    cv2.imwrite(f'{output_path}{file_name}', cv2.bitwise_and(mg, mg2))
    cv2.imwrite(f'{output_path}f_{file_name}', np.hstack([img, cv2.bitwise_and(mg, mg2)]))

if __name__ == '__main__':
    os.makedirs(binary_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(thinning_path, exist_ok=True)


    pbar = tqdm(os.listdir(input_path))
    for file_name in pbar:
        pbar.set_description(f"Processing {file_name}")
        binarization(file_name)
        # wall_seg_from_histo(file_name)
        wall_thining(file_name)