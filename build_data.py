import os
import sys
import cv2 as cv
import numpy as np
import tensorflow
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array

from Train_model import train_model

def main():
    # Helper program which takes files from data_samples and creates train_data.npy containing 40,800 samples for the model in main.py to train model
    # Labels are generated in main.py, where every 200 samples in 'out' are in a class, with 204 unique classes.
    # If modifying amount of data samples, make sure 'labels' in main.py reflects the labels you want

    current_directory = os.path.abspath(os.path.dirname(__file__))
    parent_directory = os.path.dirname(current_directory)
    sys.path.append(parent_directory)
    input_path = os.path.join(current_directory,"data_samples")
    sample_list = os.listdir(input_path)


    #build training data

    out = []
    datagen = ImageDataGenerator(
        rotation_range=60,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest'
    )

    for i in range(1,int(len(sample_list))):
        temp = np.array(cv.imread( os.path.join(current_directory,"data_samples",str(str(i) + ".bmp")), cv.IMREAD_GRAYSCALE))
        temp = cv.resize(temp, (165,230))
        temp = temp[23:109, 14:152]
        out.append(temp)
        for _ in range(199):
            arr = img_to_array(temp)
            augment_arr = datagen.random_transform(arr)
            augment_arr = augment_arr.reshape((1,) + augment_arr.shape)
            augment = array_to_img(augment_arr[0],scale=True)
            augment = np.array(augment)
            out.append(augment)

    out = np.array(out)
    print("fin: ",len(out))
    np.save('train_data.npy', out)


if __name__ == '__main__':
    main()
