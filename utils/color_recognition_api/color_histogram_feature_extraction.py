#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Mokhamad Wijaya
# --- Reference      : https://github.com/ahmetozlu/color_recognition
# --- Mail           : oskop17@gmail.com
# --- Date           : 17th June 2019
# ----------------------------------------------

from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
from utils.color_recognition_api import knn_classifier as knn_classifier


def color_histogram_of_test_image(test_src_image):

    # load the image

    chans = cv2.split(test_src_image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue
            # print(feature_data)

    with open('data/test.data', 'w') as myfile:
        myfile.write(feature_data)


def color_histogram_of_training_image(img_name):

    # detect image color by using image file name to label training data.
    # if you want to add more color, just add the "if elif" with new color
    # name like codes for each color bellow

    if 'red' in img_name:
        data_source = 'red'
    elif 'yellow' in img_name:
        data_source = 'yellow'
    elif 'green' in img_name:
        data_source = 'green'
    elif 'orange' in img_name:
        data_source = 'orange'
    elif 'white' in img_name:
        data_source = 'white'
    elif 'black' in img_name:
        data_source = 'black'
    elif 'blue' in img_name:
        data_source = 'blue'
    elif 'violet' in img_name:
        data_source = 'violet'

    # load the image
    image = cv2.imread(img_name)

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue

    with open('data/training.data', 'a') as myfile:
        myfile.write(feature_data + ',' + data_source + '\n')


def training():

    # If you want to add more color name, just write code like
    # color image extraction code section bellow with new color
    # name directory

    COLOR_DATASET_PATH = 'data/color_training_dataset/'

    # red color training images
    for f in os.listdir(COLOR_DATASET_PATH + 'red'):
        color_histogram_of_training_image(COLOR_DATASET_PATH + 'red/' + f)

    # yellow color training images
    for f in os.listdir(COLOR_DATASET_PATH + 'yellow'):
        color_histogram_of_training_image(COLOR_DATASET_PATH + 'yellow/' + f)

    # green color training images
    for f in os.listdir(COLOR_DATASET_PATH + 'green'):
        color_histogram_of_training_image(COLOR_DATASET_PATH + 'green/' + f)

    # orange color training images
    for f in os.listdir(COLOR_DATASET_PATH + 'orange'):
        color_histogram_of_training_image(COLOR_DATASET_PATH + 'orange/' + f)

    # white color training images
    for f in os.listdir(COLOR_DATASET_PATH + 'white'):
        color_histogram_of_training_image(COLOR_DATASET_PATH + 'white/' + f)

    # black color training images
    for f in os.listdir(COLOR_DATASET_PATH + 'black'):
        color_histogram_of_training_image(COLOR_DATASET_PATH + 'black/' + f)

    # blue color training images
    for f in os.listdir(COLOR_DATASET_PATH + 'blue'):
        color_histogram_of_training_image(COLOR_DATASET_PATH + 'blue/' + f)		
