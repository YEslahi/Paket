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

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse

# import utility
import preprocessing
import training
# --------------------------------------------------------------------------------------
#                                      Variables
# --------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='paket')
    

# Name of the column to be searched.
input_params = {
    'PROD_BARCODE_COL' : 'Barcode',
    'PROD_CAT_COL': 'Category',
    'PROD_SUBCAT_COL': ' Subcategory',
    'PROD_NAME_COL': 'Name',
    'img_data_path': '../data/img/all_images/',
    'mapping_data_path': '../data/csv/',
    'train_size': 0.7,
    'test_size': 0.1,
    'valid_size': 0.2
}

labels = pd.read_csv('../data/csv/label_int_mapping.csv').iloc[:,0].values
class_cnt = len(labels)

print("number of uninque labels:", class_cnt)
print("some of the labels: ", labels)

# --------------------------------------------------------------------------------------
#                                      Preprocessing
# --------------------------------------------------------------------------------------
train, test, valid = preprocessing.run_preprocessing(input_params)

# --------------------------------------------------------------------------------------
#                                      model training
# --------------------------------------------------------------------------------------
training.train(input_params, train, test, valid, class_cnt)