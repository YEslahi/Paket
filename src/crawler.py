# -*- coding: UTF-8 -*-

from decouple import config
import pandas as pd
from pathlib import Path
import os
import time
import glob
import shutil
from google_images_search import GoogleImagesSearch
# from icrawler.builtin import GoogleImageCrawler
import utility


# Inputs
# -------------------------------------------------
DEBUG = config('DEBUG', default=False, cast=bool)
# Name of the column to be searched.
PROD_BARCODE_COL = 'Barcode'
PROD_CAT_COL = 'Category'
PROD_SUBCAT_COL = ' Subcategory'
PROD_NAME_COL = 'Name'
# path
data_path = '../data/'
path = '../'
# API key, Search engine ID
gis = GoogleImagesSearch(config('API_KEY'),
                         config('API_PROJECT_CX'))

# -------------------------------------------------
def query_image(row, store_path):
    _search_params = {
        'q': row[PROD_NAME_COL],
        'num': 11,
        'fileType': 'jpg'}

    try:
    # this will search, download and resize:
        gis.search(search_params=_search_params,
                   path_to_dir=store_path,
                   width=224,
                   height=224,
                   custom_image_name='img')
    except OSError:
        print('OS error handled.')
    time.sleep(20)
    return

def copy_prof_image(row, store_path):
    """ bring the photo from professional set into the corresponding barcode folder """

    prof_img_pattern = str(int(row[PROD_BARCODE_COL])) + '_paket_*.jpg'
    _path = Path('../data/img/paket_professional/')
    for img_path in list(_path.glob(prof_img_pattern)):
        shutil.copyfile(img_path,
                        store_path+img_path.name)
    return

# read csv file
paket_products = pd.read_excel(data_path + 'csv/PaketProducts5000.xlsx')
# select desired columns
paket_products = paket_products[[PROD_BARCODE_COL, PROD_NAME_COL, PROD_CAT_COL, PROD_SUBCAT_COL ]]
print(paket_products.head())
print("Input data size before cleaning:", paket_products.shape)
print("cnt nan barcodes: ", len(paket_products[paket_products[PROD_BARCODE_COL].isna()]))
print("cnt nan names: ", len(paket_products[paket_products[PROD_NAME_COL].isna()]))
print("cnt nan categories: ", len(paket_products[paket_products[PROD_CAT_COL].isna()]))
print("cnt nan subcategories: ", len(paket_products[paket_products[PROD_SUBCAT_COL].isna()]))

paket_products.dropna(inplace=True)
print("Input data size after cleaning:", paket_products.shape)


# iterate row wise
for index, row in paket_products.iterrows():
    # make a directory for that name.
    store_path = path + 'data/img/paket_crawled/barcodes/' + str(int(row[PROD_BARCODE_COL])) + '/'
    Path(store_path).mkdir(parents=True, exist_ok=True)
    # search images if the path is emtpy.
    # search images if only prof images are available.
    if utility.dir_empty(store_path) or utility.dir_file_cnt(store_path)<5:
        print("index:", index)
        print(" row[Barcode]", int(row[PROD_BARCODE_COL]))
        print(" row[Name]", row[PROD_NAME_COL], "\n\n")
        
        # query the product name
        query_image(row, store_path)
    try:
        assert(not utility.dir_empty(store_path))
    except AssertionError:
        # wait two minute
        time.sleep(300)
        print("waiting 300 seconds before query again...")
        # query the product name
        query_image(row, store_path)
    
    # bring the photo from professional set into the corresponding barcode folder
    copy_prof_image(row, store_path)


