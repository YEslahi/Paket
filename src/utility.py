import pandas as pd
import numpy as np
import os
from pathlib import Path
import shutil
from distutils.dir_util import copy_tree
import random
import cv2
from PIL import Image
import time
import glob

# plot images
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def is_image(folder, filename, verbose=True):

    data = open(folder+filename,'rb').read(10)
    # check if file is JPG or JPEG
    if data[:3] == b'\xff\xd8\xff':
        if verbose == True:
             print(filename+" is: JPG/JPEG.")
        return True

    # check if file is PNG
    if data[:8] == b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a':
        if verbose == True:
             print(filename+" is: PNG.")
        return True

    # check if file is GIF
    if data[:6] in [b'\x47\x49\x46\x38\x37\x61', b'\x47\x49\x46\x38\x39\x61']:
        if verbose == True:
             print(filename+" is: GIF.")
        return True

    return False
def validate_image_type(folder = '../data/img/all_images/'):
    """get a folder with images and check whether they are from type JPG/JPEG, PNG, or GIF.
    if not, the photo will be removed from the directory!"""
    
    # go through all files in desired folder
    for filename in os.listdir(folder):
         # check if file is actually an image file
         if is_image(folder, filename, verbose=True) == False:
              # if the file is not valid, remove it
              print("Warning! your photo is being removed from the directory!")
              import code; code.interact(local=dict(globals(), **locals()))
              os.remove(os. path. join(folder, filename))

# validate image files
def validate_image_file(folder_path):
    '''
        remove the images that have loading problems during
        folder_path: 
            path to the training files directory.
    '''
    extensions = []
    for fldr in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, fldr)
        for filee in os.listdir(sub_folder_path):
            file_path = os.path.join(sub_folder_path, filee)
            print('** Path: {}  **'.format(file_path), end="\r", flush=True)
            try:
                im = Image.open(file_path)
                rgb_im = im.convert('RGB')
                if filee.split('.')[1] not in extensions:
                    extensions.append(filee.split('.')[1])
            except UnidentifiedImageError:
                print("remove image in ", file_path)
                os.remove(file_path)


def reset_img_name(paket_products, input_params, _path='../data/img/paket_crawled/barcodes/'):
    """ iterates over the full image folders, sets the name in format
    Barcode_imgN_Paket.JPG. 
    """
    map_img_label = []
    for path in Path(_path).iterdir():
        for index, fileinpath in enumerate(Path(path).iterdir()):
            if fileinpath.is_file():
                barcode = path.stem
                extension = fileinpath.suffix
                directory = fileinpath.parent
                new_name = barcode + "_paket_" + str(index) + extension
                fileinpath.rename(Path(directory, new_name))
                import code; code.interact(local=dict(globals(), **locals()))
                try:
                    map_img_label.append([new_name,
                     paket_products[paket_products[input_params['PROD_BARCODE_COL']] == int(barcode)][input_params['PROD_SUBCAT_COL']].item()])
                except ValueError:
                    print("barcode: ", barcode, "not found!" )
    
    pd.DataFrame(map_img_label, columns=['image', 'label']). \
        to_csv('../data/csv/map_img_label.csv', index=False)

    return

def store_img_label(paket_products, input_params, _path='../data/csv/'):
    map_img_label = pd.DataFrame(columns=['image', 'label'])
    # for path in Path(_path).iterdir():
    return

def all_file_classes_unify(paket_products, input_params, PATH = '../'):
    # create a folder for each subcategory. 
    # put the corresponding barcodes in that directory
    DATA_PATH = PATH + 'data/img/paket_crawled/barcodes/'
    for index, row in paket_products.iterrows():
        store_path = PATH + 'data/img/classes/' + str(row[input_params['PROD_SUBCAT_COL']]) + '/'
        # Path(store_path).mkdir(parents=True, exist_ok=True)

        # copy all the barcode folders of the subcategory to this directory
        copy_tree(DATA_PATH + str(int(row[input_params['PROD_BARCODE_COL']])),
                        PATH + 'data/img/classes/' + str(row[input_params['PROD_SUBCAT_COL']]))
                        # dirs_exist_ok=True)  # dir exist ok in python3.8+
    return


def func_3(paket_products):
    # create one folder with all images inside. 
    for subcategory in paket_products.Subcategory.dropna():
        shutil.copytree(PATH + 'data/img/classes/' + str(subcategory) + '/',
                        PATH + 'data/img/all_images/',
                        dirs_exist_ok=True)
    return


def file_lines_to_list(path):
    '''
    ### Convert Lines in TXT File to List ###
    path: path to file
    '''
    with open(path) as f:
        content = f.readlines()
    content = [(x.strip()).split() for x in content]
    return content


def get_file_name(path):
    '''
    ### Get Filename of Filepath ###
    path: path to file
    '''
    basename = os.path.basename(path)
    onlyname = os.path.splitext(basename)[0]
    return onlyname

def dir_empty(dir_path):
    try:
        return not any([True for _ in os.scandir(dir_path)])
    except FileNotFoundError:
        return True

def dir_file_cnt(dir_path):
    path, dirs, files = next(os.walk(dir_path))
    file_count = len(files)
    return file_count



# # another validaate type start        
# #-------------------------------
# # TOdo: clean this up and make it a function.
# from struct import unpack
# import os
# # from tqdm import tqdm
# import os.path as osp

# folder = '../data/img/all_images/'
# images = os.listdir(folder)
# marker_mapping = {
#     0xffd8: "Start of Image",
#     0xffe0: "Application Default Header",
#     0xffdb: "Quantization Table",
#     0xffc0: "Start of Frame",
#     0xffc4: "Define Huffman Table",
#     0xffda: "Start of Scan",
#     0xffd9: "End of Image"
# }


# class JPEG:
#     def __init__(self, image_file):
#         with open(image_file, 'rb') as f:
#             self.img_data = f.read()
    
#     def decode(self):
#         data = self.img_data
#         while(True):
#             marker, = unpack(">H", data[0:2])
#             # print(marker_mapping.get(marker))
#             if marker == 0xffd8:
#                 data = data[2:]
#             elif marker == 0xffd9:
#                 return
#             elif marker == 0xffda:
#                 data = data[-2:]
#             else:
#                 lenchunk, = unpack(">H", data[2:4])
#                 data = data[2+lenchunk:]            
#             if len(data)==0:
#                 break        


# bads = []

# for img in tqdm(images):
#   image = osp.join(folder,img)
#   image = JPEG(image) 
#   try:
#     image.decode()   
#   except:
#     bads.append(img)


# for name in bads:
#     import code; code.interact(local=dict(globals(), **locals()))
#     os.remove(osp.join(folder, name))
# # another validaate type end
# #-------------------------------