import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main():
    # Main program for users to add/remove cards and generate report of their collection
    current_directory = os.path.abspath(os.path.dirname(__file__))

    parent_directory = os.path.dirname(current_directory)
    # Add the parent directory to sys.path
    sys.path.append(parent_directory)

    if os.path.exists('Collection.npy') == False:
        print("Card collection 'collection.npy' not found, creating new card collection...")
        collection = np.zeros(204,np.uint8)
        np.save('Collection.npy', collection)
    else:
        print("Card collection 'collection.npy' found, loading collection...")
        collection = np.load('Collection.npy')

    labels = list(range(1, 205))

    out = []

    for i in range(1,205):
        temp = np.array(cv.imread( os.path.join(current_directory,"data_samples",str(str(i) + ".bmp")), cv.IMREAD_GRAYSCALE))
        temp = cv.resize(temp, (165,230))
        out.append(temp)

    plt.figure()
    plt.imshow(out[203], 'gray')
    plt.show()


    out = np.array(out)
    np.save('train_data.npy', out)

    #train_data = np.load('train_data.npy')
    #plt.figure()
    #plt.imshow(train_data[203])
    #plt.show()



if __name__ == '__main__':
    main()
