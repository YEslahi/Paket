# Credit: https://github.com/tranleanh/data-augmentation
import utility
import os
import cv2
import numpy as np
import random
import tensorflow as tf
import datetime
from PIL import Image

def run_augmentation(methods, image, n=10, scale=0.5):
    '''
    ### Uses the list of specified augmentation methods. 
    Parameters:
    methods: list, default=['random_crop']
             'random_crop': randomly crop the image and scale it to org size.
             'cut_out'
             'color_jitter'
             'noise_adder'
             'filters'
             'all'
    '''
    # TODO: use it to call the additional augmentations in later versions.
    
def normalize(image, label):
    print("in normalization.")
    normalized_image = (image - 127.5) / 127.5
    # normalized_image = image
    return normalized_image, label

def resize(image, label):
    # data augmentation here.
    # resize the data into 224,224,3s

    resized_image = tf.image.resize(image, [224, 224]) 
    return resized_image, label

def _randomcrop(img):
    '''
    ### Random Crop ###
    Parameters: 
        img: image
        scale: float, random value between 0.75 - 1
            percentage of cropped area
    Return:
        resized: list
                new images genenrated from the org img.
    '''
    # Crop image
    scale = random.uniform(0.75, 1)
    height, width = int(img.shape[0] * scale), int(img.shape[1] * scale)
    
    x = random.randint(0, img.shape[0] - int(width))
    y = random.randint(0, img.shape[1] - int(height))
    cropped = img[y:y+height, x:x+width]
    resized = tf.image.resize(cropped, [img.shape[1], img.shape[0]])
    return resized

def random_crop(image, label):
    # augment data
    random_cropped_image = _randomcrop(image)
    return random_cropped_image, label
