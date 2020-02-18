import numpy as np
import cv2
import random
import os
import glob
import csv

# calculate means and std
train_txt_path = 'D:/phython_work/cifar/cifar-10-batches-py/train_canny/train.csv'

means = 0
stdevs = 0

index = 1
num_imgs = 0
with open(train_txt_path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        path = row[0]
        index += 1

        num_imgs += 1
        img = cv2.imread(path)
        img = img.astype(np.float32) / 255.

        means += img[:, :].mean()
        stdevs += img[:, :].std()

print(num_imgs)
means = np.asarray(means) / num_imgs
stdevs = np.asarray(stdevs) / num_imgs

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))