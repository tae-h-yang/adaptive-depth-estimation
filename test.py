import cv2
import torch
import sys
import os
import importlib

raw_img = cv2.imread('datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/rgb/1305031452.791720.png')
print(raw_img.shape)
cv2.imshow("", raw_img)
cv2.waitKey(0)