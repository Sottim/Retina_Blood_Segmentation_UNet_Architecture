""" Data Preprocessing :
1. Load the dataset
2. Apply data augumentation 
"""

import os # used to join paths and create some direcotory
import numpy as np 
import cv2 # read, resize and save the images
from glob import glob #extract the images from the respective folders
from tqdm import tqdm #Progress bar
import imageio #Used to read the .gif masks which we have
from albumentations import HorizontalFlip, VerticalFlip, VerticalFlip, Rotate #Data agumentations

""" Create a new directory to store the augumented data"""
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path):
    train_x = sorted(glob(os.path.join(path, "training", "images", "*.tif"))) # Training images
    train_y = sorted(glob(os.path.join(path, "training", "1st_manual", "*.gif"))) #Training masks

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.tif"))) # Testing images
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif"))) #Testing masks

    return (train_x, train_y), (test_x, test_y)

def augment_data(images, masks, save_path, augment = True):
    size = (512, 512) #size of the augumented images


    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the name from 'x' which is image and 'y' which is a mask for that image"""
        name = x.split("/")[-1].split(".")[0]

        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR) #Read as RGB image
        y = imageio.mimread(y)[0]

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            # Update the X and Y to contain the augumented images
            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]


        #If augumentation is set to false, still need to resize the images to 512 to make same throughout
        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1

if __name__ == "__main__":
    """"Seeding"""
    np.random.seed(42)

    """"Load the data"""
    data_path = r"/home/santosh/UNET_Retina_Blood_Segmentation/archive/DRIVE"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Training data:{len(train_x)} - {len(train_y)}")
    print(f"Testing data:{len(test_x)} - {len(test_y)}")

    """ Create directories to save the augmented data """
    directories = ["new_data/train/image/", "new_data/train/mask/", "new_data/test/image/", "new_data/test/mask/"]
    for directory in directories:
        create_dir(directory)

    """ Data augmentation """
    augment_data(train_x, train_y, "new_data/train/", augment=True)
    augment_data(test_x, test_y, "new_data/test/", augment=False)



