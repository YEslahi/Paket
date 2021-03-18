import augment
from pathlib import Path
import os
import cv2
import pandas as pd
def read_image():
    # TODO
    return

def store_augmented_images(augmented_images, base_img_dir, output_dir):

    for index, augmented_image in enumerate(augmented_images):
        base_img_name = base_img_dir.stem
        new_img_dir = f"{output_dir}/{base_img_name}_aug{index}.jpg"
        cv2.imwrite(new_img_dir, augmented_image)
    return new_img_dir


def run_preprocessing():
    # go through all images in train set and apply augmentation.
    map_img_label = []
    train_dir = f"../data/img/splits/train"
    # Load Images
    for path in Path(train_dir).iterdir():
        # for each directory in train(class names), create a 
        # directory in augmentation output directory.
        if path.is_dir():
            output_dir = f"../data/img/train_aug_output/{path.stem}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for index, fileinpath in enumerate(Path(path).iterdir()):
                if fileinpath.is_file():
                    img_name = fileinpath.stem
                    image = cv2.imread(str(fileinpath))
                    if image is not None:                        
                        # run augmentation
                        augmented_images = augment.run_augmentation(['random_crop'], image, n=10, scale=0.5)

                        # store augmentation
                        new_img_dir = store_augmented_images(augmented_images, fileinpath, output_dir)

                        # store the class as well as file name in a mapping file.
                        try:
                            map_img_label.append([new_img_dir, path.stem])
                        except ValueError:
                            print("Value error!")
                            import code; code.interact(local=dict(globals(), **locals()))
    
    pd.DataFrame(map_img_label, columns=['image', 'label']). \
    to_csv('../data/csv/map_img_label.csv', index=True)
                

    print("Preprocessing Done!")
