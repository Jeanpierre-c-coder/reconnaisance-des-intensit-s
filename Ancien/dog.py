# External lib
import itertools
import numpy as np
import math as math
import random as random
import time
import csv
import os
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn
from sklearn import metrics
from skimage.transform import rescale, resize, downscale_local_mean
import skimage.measure
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter1d
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray
from skimage import color
from skimage import io
import skimage.measure

## Visual Chain

"""
Function use to make a competition on an input gradiant image to construct PoI detector.
In this algo, we get each max of a gradiant image and erase around all the other PoI (competition)
params:
* input_image : input image (1D matrix)
* raidius_ecart : radius of the competition between each max
* poi_number : number of PoI extracted from the input image
"""
def compet_ptc_image(input_image, radius_ecart, poi_number = 30) :

    output_image = np.zeros_like(input_image)

    for cpt in range(poi_number):
        # we get the max value
        index = np.unravel_index(np.argmax(input_image, axis=None), input_image.shape)

        # We store the max in a image
        output_image[index[0]][index[1]] = input_image[index[0]][index[1]]

        # We remove the point and around
        input_image[index[0]-radius_ecart:index[0]+radius_ecart, index[1]-radius_ecart:index[1]+radius_ecart] = -1
    return output_image

"""
Function use to extract the keypoint from a competition image in a array.
The algo only extract the max from the input competition image
params :
* image : input compte image
* keypoint_nbr : number of keypoint to extract
@see compet_ptc_image
"""
def extract_keypoint_promethe(image, keypoint_nbr) :
    # Selection
    keypoint = []
    for i in range(keypoint_nbr) :
        indice = np.argmax(image)
        index = np.unravel_index(np.argmax(image), image.shape)

        keypoint.append([index[1], index[0]])
        image[index[0]][index[1]] = -1.0

    return keypoint

"""
Function use to build a Dog filter (difference of gaussian)
Extracted from Promethe (init_masque_pt_carac)
params :
* l = size of one border of the dog
* Theta1 : first theta use in the computation
* Theta2 : second theta use in the computation
"""
def generate_DoG_filter(l, theta2, theta1) :

    tableau = np.zeros((2*l, 2*l))

    a3 = (2. * theta1 * theta1)
    a4 = (2. * theta2 * theta2)
    a1 = 1. / (2. * math.pi * theta1 * theta1);
    a2 = 1. / (2. * math.pi * theta2 * theta2);

    for j in range(-l+1, l) :
        for i in range(-l+1, l) :
            d = i * i + j * j
            d1 = np.exp(-d / a3);
            d2 = np.exp(-d / a4);
            tableau[l + i][l + j] = ((a1 * d1 - a2 * d2));

    return tableau

"""
Function use to transform an input image into gradiant image by transform the input image into a gradiant image with a deriche filter and by convoluate the input image with a DoG filter.
The deriche is the deriche used by opencv.
The convolution is made with an fft
params :
* deriche_alpha : Alpha parameter of the deriche that control the size of the smoothing
* deriche gamma : see cv2.ximgproc.GradientDericheX
* filter_ : input filter -> generaly a DoG
@see generate_DoG_filter
"""
def process_dog_from_image(img, deriche_alpha = 0.5, deriche_gamma = 0.2,
    filter_ = None, filter_exclusion_radius = 32) :

    # Deriche grad
    img = np.array(img, dtype=np.float32)
    DericheX = cv2.ximgproc.GradientDericheX(img, deriche_alpha, deriche_gamma)
    DericheY = cv2.ximgproc.GradientDericheY(img, deriche_alpha, deriche_gamma)
    grad = np.sqrt(DericheX**2 + DericheY**2)

    # DoG convolution
    Dog_image = np.fft.fftshift(np.fft.fft2(grad.reshape(img.shape[0], img.shape[1]).copy())) * filter_
    Dog_image = np.fft.ifft2(np.fft.ifftshift(Dog_image)).real
    Dog_image[0:filter_exclusion_radius, :] = 0
    Dog_image[-filter_exclusion_radius:, :] = 0
    Dog_image[:,0:filter_exclusion_radius] = 0
    Dog_image[:,-filter_exclusion_radius:] = 0

    return Dog_image

## Data treatment
"""
\interface
Encoding of input PoI
"""
def LPMP_encode_poi(self, img_gray, tmp_keypoints,
       global_vignette_size = 54, LPMP_radius = 60
        ) :

    # Encoding of PoI
    log_polars = []
    for cpt_poi, tmp_poi in enumerate(tmp_keypoints) :
        log_polars.append(
            cv2.warpPolar(img_gray,
                (global_vignette_size,global_vignette_size),
                (tmp_poi[0],tmp_poi[1]),
                    LPMP_radius,
                    cv2.WARP_FILL_OUTLIERS
                )
            )
    return log_polars
