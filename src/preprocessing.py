import augment
from pathlib import Path
import os
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import utility

def read_image(image_file, label):

    image = tf.io.read_file('../data/img/all_images/'+ image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    return image, label

def get_splits_from_file(input_params):
    #TODO: don't hardcoded image_name
    all_img_names = pd.read_csv(input_params['mapping_data_path']+'map_img_label.csv')
    all_img_names = all_img_names.sample(frac=1)
    train, valid, test = \
        np.split(all_img_names,
                [   int(input_params['train_size']*len(all_img_names)),
                    int((input_params['valid_size'] + input_params['train_size'])*len(all_img_names))
                ])
    if utility.dir_empty(input_params['mapping_data_path'] + 'splits/'):
        train.to_csv(input_params['mapping_data_path'] + 'splits/train.csv', index=False)
        test.to_csv(input_params['mapping_data_path'] + 'splits/test.csv', index=False)
        valid.to_csv(input_params['mapping_data_path']+'splits/valid.csv', index=False)
    else:
        train = pd.read_csv(input_params['mapping_data_path'] + 'splits/train.csv')
        test = pd.read_csv(input_params['mapping_data_path'] + 'splits/test.csv')
        valid = pd.read_csv(input_params['mapping_data_path'] + 'splits/valid.csv')
    
    train = tf.data.Dataset.from_tensor_slices((tf.constant(train['image_name'].values), tf.constant(train['label'].values)))
    test = tf.data.Dataset.from_tensor_slices((tf.constant(test['image_name'].values), tf.constant(test['label'].values)))
    valid = tf.data.Dataset.from_tensor_slices((tf.constant(valid['image_name'].values), tf.constant(valid['label'].values)))

    return train, test, valid
    
def run_preprocessing(input_params):
    """
        Performs: 1. train/test/valid split files.
                  2. Calls Augmentation
        Parameters:
            input_params: dict
        Return : 
            None
    """
    # todo: do the resize before splits
    train, test, valid = get_splits_from_file(input_params)
    train = train.map(read_image).map(augment.resize).map(augment.normalize).map(augment.random_crop).shuffle(buffer_size=1024).batch(64)
    test = test.map(read_image).map(augment.resize).batch(64)
    valid = valid.map(read_image).map(augment.resize).batch(64)
    print("Preprocessing Done!")

    return train, test, valid

