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

    '''out = []

    for i in range(1,205):
        temp = np.array(cv.imread( os.path.join(current_directory,"data_samples",str(str(i) + ".bmp")), cv.IMREAD_GRAYSCALE))
        temp = cv.resize(temp, (165,230))
        out.append(temp)

    plt.figure()
    plt.imshow(out[203], 'gray')
    plt.show()


    out = np.array(out)
    np.save('train_data.npy', out)

    train_data = np.load('train_data.npy')
    plt.figure()
    plt.imshow(train_data[0])
    plt.show()'''

    base = np.array(cv.cvtColor(cv.imread( os.path.join(current_directory,"data_samples","testcard5.bmp"), cv.COLOR_RGB2BGR), cv.COLOR_BGR2RGB))

    top_row, bottom_row, width, center = get_card_dimension(base, 9)

    if(bottom_row == len(base)):
        top_row, bottom_row, width, center = get_card_dimension(base, 2)
    
    


def get_card_dimension(img, kernel_size):
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    lower = np.array([1,1,1])
    upper = np.array([255,50,255])
    mask = cv.inRange(hsv, lower, upper)
    erodeKernel = np.ones((kernel_size,kernel_size),np.uint8)
    mask = cv.dilate(mask,erodeKernel)

    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=4)
    max_label, max_size = max([(i, stats[i,cv.CC_STAT_AREA]) for i in range(1, nb_components)], key= lambda x:x[1])
    mask[output != max_label] = 0

    top_row = int(stats[max_label,cv.CC_STAT_TOP])
    left_column = int(stats[max_label,cv.CC_STAT_LEFT])
    bottom_row = top_row + int(stats[max_label,cv.CC_STAT_HEIGHT])
    right_column = left_column + int(stats[max_label,cv.CC_STAT_WIDTH])
    width = int((bottom_row - top_row) / 1.4)

    
    print("top: ",top_row)
    print("left: ",left_column)
    print("bottom: ",bottom_row)
    print("right: ",right_column)
    print("size: ", (bottom_row - top_row), " x ", width)
    print("image width: ", len(mask))
    print("center: ", centroids[0])

    plt.figure()
    plt.imshow( mask)
    plt.show()

    return top_row, bottom_row, width, centroids[0]


if __name__ == '__main__':
    main()
