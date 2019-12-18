#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : ML_Project_2
# @Author       : Wei Jiang, Xiaoyu Lin, Yao Di
# @File         : detection.py
# @Discription  : Run this file can apply trained model on the test dataset in dateset/test/frame, and generate a mask
#                 sequence for corresponding frames.
#                 You can change the size of dilation by changing tha parameter in dilate() function. Please keep the
#                 size of dilation the same as the size of erosion in yeastcell.py
import skimage
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import yeastcell
import tensorflow as tf

################################################
# Root
################################################
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Directory to save logs and trained model
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
WEIGHT_DIR = os.path.join(ROOT_DIR, "model")
DETECTION_DIR = os.path.abspath("../dataset/test/frame")

################################################
# Preferences
################################################
# Inference Configuration
config = yeastcell.YeastCellConfig()

################################################
# Preferences
################################################
# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/device:CPU:0"  # /device:CPU:0 or /device:GPU:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# Only inference mode is supported right now
TEST_MODE = "inference"

################################################
# Load Detection Dataset
################################################
dataset = yeastcell.YeastCellDataset()
dataset.load_cells(DETECTION_DIR)
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

################################################
# Load Load Model
################################################
# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=LOGS_DIR,
                              config=config)
# Path to a specific weights file
weights_path = os.path.join(WEIGHT_DIR, "mask_rcnn_yeast.h5")
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

################################################
# Run Detection
################################################
final_mask = []
image_ids = dataset.image_ids
for image_id in image_ids:
    image = dataset.load_image(image_id)
    # Run detection
    results = model.detect([image], verbose=1)
    # image = dataset.load_original_image(image_id)
    class_names = ['BG', 'cell']
    r = results[0]
    # Dilate the masks
    for i in range(r['masks'].shape[2]):
        r['masks'][:, :, i] = yeastcell.dilate(r['masks'][:, :, i], size=3)
    # Visualize results
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'], show_bbox=False, show_label=False)
    # Change make stack to single 2D mask
    mask = np.zeros(r['masks'][:, :, 0].shape)
    for i in range(r['masks'].shape[2]):
        mask = mask + r['masks'][:, :, i] * (i + 1)
    final_mask.append(mask)

# Generate results.tif file
final_mask = np.array(final_mask)
final_mask = final_mask.astype('int16')
skimage.external.tifffile.imsave('../dataset/test/mask/results.tif', final_mask, imagej=True)
