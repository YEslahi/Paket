# Credit: https://github.com/tranleanh/data-augmentation
import utility
import os
import cv2
import numpy as np
import random


def randomcrop(img, n=10, scale=0.5):
    '''
    ### Random Crop ###
    Parameters:
        img: image
        scale: float, default=0.5
            percentage of cropped area
        n:  int, default=10
            number of photos to create from each org img.
    Return:
        augmented_images: list
                new images genenrated from the org img.
    '''
    #TODO check if there is padding for crop outside of the image.
    #TODO is it better to zoom and scale to the size we want or better to add padding or both?
    # Crop image
    height, width = int(img.shape[0] * scale), int(img.shape[1] * scale)
    augmented_images = []
    for i in range(n):
        x = random.randint(0, img.shape[1] - int(width))
        y = random.randint(0, img.shape[0] - int(height))
        cropped = img[y:y+height, x:x+width]
        resized = cv2.resize(cropped, (img.shape[1], img.shape[0]))
        augmented_images.append(resized)
    return augmented_images

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
    # TODO
    for method in methods:
        if method == 'random_crop':
            return  randomcrop(image, n, scale)

