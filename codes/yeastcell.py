#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : ML_Project_2
# @Author       : 
# @File         : yeastcell.py
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

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils

class YeastCellConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "YeastCell"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + cell

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 20


class YeastCellDataset(utils.Dataset):

    def load_cells(self, dataset_dir):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("YeastCell", 1, "cell")

        # Add images
        frames_ids = next(os.walk(dataset_dir))[2]

        for frames_id in frames_ids:
            frames_id = '.'.join(frames_id.split('.')[:-1])
            framesize = skimage.io.imread(os.path.join(dataset_dir, "{}.tif".format(frames_id))).shape[0]

            if frames_id == 'michael_1.2_im':
                framesize -= 1
            if frames_id == 'michael_1.2.2_im':
                framesize -= 2

            for i in range(framesize):
                self.add_image(
                    "YeastCell",
                    image_id=frames_id + "*" + str(i),
                    path=os.path.join(dataset_dir, "{}.tif".format(frames_id)))

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image_name = self.image_info[image_id]['id']
        print(image_name)

        frame_number = self.image_info[image_id]['id'].split('*')[1]
        image = skimage.io.imread(self.image_info[image_id]['path'])[int(frame_number)]
        image = image.astype('int16')###
        image = image / image.max() * 255
        # # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "mask")

        # # Read mask files from .tiff image
        mask = []
        name = info['id'].split('*')[0]
        name = name.replace('frames', 'mask')
        name = name.replace('im', 'mask')
        frame_number = info['id'].split('*')[1]
        m = skimage.io.imread(os.path.join(mask_dir, "{}.tif".format(name)))[int(frame_number)]
        m = m.astype('int16')###
        instance_number = m.max()
        if instance_number == 0:
            print("Frame " + name + ' ' + str(frame_number) + " have no cell!")

        for i in range(1, instance_number+1):
            m_instance = np.where(m != i, 0, m)
            m_instance = m_instance.astype(np.bool)
            mask.append(m_instance)
        mask = np.stack(mask, axis=-1)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "YeastCell":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)

def erode(images, random_state, parents, hooks):
    # Taking a matrix of size 3 as the kernel
    size = 3
    kernel = np.ones((size, size), np.uint8)
    center = radius = (size-1)/2
    for x in range(size):
        for y in range(size):
            if (x-center)**2 + (y-center)**2 > radius**2:
                kernel[x][y] = 0

    img_erosion = []
    for image in images:
        img_erosion.append(cv2.erode(image, kernel, iterations=1))

    return img_erosion

def dilate(image, size=5):
    # Taking a matrix of size 3 as the kernel
    kernel = np.ones((size, size), np.uint8)
    center = radius = (size-1)/2
    for x in range(size):
        for y in range(size):
            if (x-center)**2 + (y-center)**2 > radius**2:
                kernel[x][y] = 0

    image = image.astype('int16')
    image = cv2.UMat(image)
    img_dilation = cv2.dilate(image, kernel, iterations=1)
    img_dilation = img_dilation.get()
    return img_dilation
