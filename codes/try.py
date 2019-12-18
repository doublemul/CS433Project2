#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : ML_Project_2
# @Author       : 
# @File         : try.py
# @Discription  :
import skimage
from imgaug import augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import yeastcell
import tifffile
from yeastSegHelpers import tile_helpers

images = skimage.io.imread('../dataset/train/frame/augoustina_first_im.tif')
image = images[1]
tiled = tile_helpers.tile_image(image, 1024)
print(np.array(tiled).shape)
# if images.ndim != 3:
#     images = skimage.color.gray2rgb(images)
# print(images.shape)
#
# augmentation = iaa.Lambda(func_images=yeastcell.erode)
# print('imageshape' + str(images.shape))
# image_aug = augmentation(images=images)
# # print(image_aug.shape)
# lis = [[1, 3, 4],[2,4,5]]
# print(len(lis))
# mask =[]
# a = np.array([[1, 3, 4], [3, 5, 6], [7, 8, 9], [7, 8, 9]])
# b = np.array([[4, 3, 4], [3, 7, 6], [7, 4, 9], [7, 8, 9]])
#
# mask.append(a)
# mask.append(b)
# mask = np.array(mask)
# mask = mask.astype('int16')
# # tifffile.imwrite('test.tif', mask, shape=mask.shape,dtype=np.int8)
# skimage.external.tifffile.imsave('test.tif', mask, imagej=True)