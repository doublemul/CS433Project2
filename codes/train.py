#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : ML_Project_2
# @Author       : Xiaoyu Lin
# @File         : train.py
# @Discription  :

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
from imgaug import augmenters as iaa

################################################
# Device
################################################
# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

DEVICE = "/device:CPU:0"  # /device:CPU:0 or /device:GPU:0

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
TRAIN_DIR = os.path.abspath("../dataset/train/frame")
VALIDATION_DIR = os.path.abspath("../dataset/validation/frame")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'model', "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

################################################
# Configurations
################################################

config = yeastcell.YeastCellConfig()

################################################
# Dataset
################################################

# Training dataset
dataset_train = yeastcell.YeastCellDataset()
dataset_train.load_cells(TRAIN_DIR)
dataset_train.prepare()

# Validation dataset
dataset_val = yeastcell.YeastCellDataset()
dataset_val.load_cells(VALIDATION_DIR)
dataset_val.prepare()

# print train dataset information
print('Train dataset information:')
print("Image Count: {}".format(len(dataset_train.image_ids)))
print("Class Count: {}".format(dataset_train.num_classes))
for i, info in enumerate(dataset_train.class_info):
    print("{:3}. {:50}".format(i, info['name']))

# print validation dataset information
print('Validation dataset information:')
print("Image Count: {}".format(len(dataset_val.image_ids)))
print("Class Count: {}".format(dataset_val.num_classes))
for i, info in enumerate(dataset_val.class_info):
    print("{:3}. {:50}".format(i, info['name']))

# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# image_ids = dataset_val.image_ids
# for image_id in image_ids:
#     print(image_id)
#     image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
#             dataset_val, config, image_id, use_mini_mask=False)
#     # log("molded_image", image)
#     # log("mask", mask)
#     # visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names,
#     #                             show_bbox=False)

################################################
# Create Model
################################################
with tf.device(DEVICE):
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

# Choose weights to start with
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

################################################
# Training
################################################

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads')

augmentation = iaa.Sequential([
    iaa.Lambda(func_images=yeastcell.erode),
    iaa.OneOf([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Sometimes(0.5, iaa.Sequential([
            iaa.Fliplr(1.0),
            iaa.Fliplr(1.0)]))])
])

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=10,
            augmentation=augmentation,
            layers="all")

# Save weights
model_path = os.path.join(ROOT_DIR, 'model', "mask_rcnn_yeast.h5")
model.keras_model.save_weights(model_path)