import pandas as pd
from pathlib import Path
import os
import time
from google_images_search import GoogleImagesSearch
import random
import shutil
import itertools
import matplotlib.pyplot as plt
import scipy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from PIL import ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import MobileNet as MOBILENET
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import cv2
import numpy as np

import utility
import preprocessing

# Name of the column to be searched.
input_params = {
    'PROD_BARCODE_COL' : 'Barcode',
    'PROD_CAT_COL': 'Category',
    'PROD_SUBCAT_COL': ' Subcategory',
    'PROD_NAME_COL': 'Name'
}
# path
PATH = '../'


paket_products = pd.read_excel(PATH + 'data/csv/PaketProducts5000.xlsx')

paket_products = paket_products[[input_params['PROD_BARCODE_COL'],
                                 input_params['PROD_NAME_COL'],
                                 input_params['PROD_CAT_COL'],
                                 input_params['PROD_SUBCAT_COL']]]
paket_products.dropna(inplace=True)

labels = list(paket_products[input_params['PROD_SUBCAT_COL']].unique())
# to avoid problem with persian characters, use int for them.
# key: int, value: label in persian
labels = {str(key): str(value) for key, value in enumerate(labels)}

# store the mapping to csv
pd.DataFrame.from_dict(labels, orient='index').to_csv("../data/csv/label_int_mapping.csv", index=True)
class_cnt = len(labels)

print("number of uninque labels:", class_cnt)
print("some of the labels: ", labels)

# --------------------------------------------------------------------------------------
#                                      Preprocessing
# --------------------------------------------------------------------------------------
# preprocessing.run_preprocessing()

# set data path
train_path =  '../data/img/train_aug_output/'
test_path = '../data/img/splits/test/'
valid_path = '../data/img/splits/valid/'

print("In train...")
train_batches = ImageDataGenerator() \
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=labels.keys(), batch_size=200)

print("In validation...")
valid_batches = ImageDataGenerator() \
    .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=labels.keys(), batch_size=200)

print("In test...")
test_batches = ImageDataGenerator() \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=labels.keys(), batch_size=200, shuffle=False)

# --------------------------------------------------------------------------------------
#                                      model training
# --------------------------------------------------------------------------------------
mobilenet = MOBILENET(include_top=False,
                      input_shape=(224, 224, 3),
                      weights='imagenet',
                      pooling='avg',
                      dropout=0.001)
mobilenet.summary()
# select till which layer use mobilenet.
base_model = Model(inputs=mobilenet.input, outputs=mobilenet.output)
base_model.summary()
import code; code.interact(local=dict(globals(), **locals()))
model = Sequential([
    base_model,
    Dropout(0.2),
    Dense(units=class_cnt, activation='softmax'),
])
model.layers[0].trainable = False
model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

try:
    model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)
except Exception:
    # TODO
    print("UnidentifiedImageError!")

# prediction
test_imgs, test_labels = next(test_batches)
# utility.plot_images(test_imgs)
# print(test_labels)

predictions = model.predict(x=test_batches, verbose=0)

results = model.evaluate(test_batches)
