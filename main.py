import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from Train_model import train_model

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

    #labels = list(range(1, 205))
    labels = []
    for i in range(1,205):
        for j in range(1,11):
            labels.append(i)

    print("len: ",len(labels))

    #build training data

    '''out = []
    datagen = ImageDataGenerator(
        rotation_range=60,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest'
    )

    for i in range(1,205):
        temp = np.array(cv.imread( os.path.join(current_directory,"data_samples",str(str(i) + ".bmp")), cv.IMREAD_GRAYSCALE))
        #orig 660 x 920
        temp = cv.resize(temp, (165,230))
        temp = temp[23:109, 14:152]
        #plt.figure()
        #plt.imshow(temp)
        #plt.show()
        out.append(temp)
        for _ in range(9):
            arr = img_to_array(temp)
            augment_arr = datagen.random_transform(arr)
            augment_arr = augment_arr.reshape((1,) + augment_arr.shape)
            augment = array_to_img(augment_arr[0],scale=True)
            augment = np.array(augment)
            out.append(augment)

    print("len: ",len(out))

    plt.figure()
    plt.imshow(out[203], 'gray')
    plt.show()


    out = np.array(out)
    np.save('train_data.npy', out)

    train_data = np.load('train_data.npy')
    plt.figure()
    plt.imshow(train_data[17])
    plt.show()'''

    #test get_card_dimension

    '''base = np.array(cv.cvtColor(cv.imread( os.path.join(current_directory,"data_samples","testcards.bmp"), cv.COLOR_RGB2BGR), cv.COLOR_BGR2RGB))

    top_row, bottom_row, width, center = get_card_dimension(base, 9)

    if(bottom_row == len(base)):
        top_row, bottom_row, width, center = get_card_dimension(base, 2)'''
    
    #train model and test for image 18.bmp
    
    train_data = np.load('train_data.npy')
    labels = np.array(labels)
    
    if os.path.exists('card_model.keras') == False:
        print("CNN model card_models.keras not found, creating and training new model")
        model, history = train_model(train_data,labels)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()
        model.save('card_model.keras')
    else:
        model = keras.models.load_model('card_model.keras')


    base = np.array(cv.cvtColor(cv.imread( os.path.join(current_directory,"data_samples","testcard7.bmp"), cv.COLOR_RGB2BGR), cv.COLOR_BGR2RGB))

    top_row, bottom_row, width, center = get_card_dimension(base, 9)

    if(bottom_row == len(base)):
        top_row, bottom_row, width, center = get_card_dimension(base, 2)

    capped_left = max(int(center[0] - width/2), 0)
    capped_right = min(len(base[0]),int(center[0] + width/2))

    window = cv.cvtColor(base[top_row:bottom_row,capped_left:capped_right], cv.COLOR_RGB2GRAY)
    window = cv.resize(window, (165,230))
    window = window[23:109, 14:152]
    train_data = np.load('train_data.npy')
    plt.figure()
    plt.imshow(train_data[170])
    plt.show()
    plt.figure()
    plt.imshow(window)
    plt.show()
    #h = bottom_row - top_row
    #w = capped_right - capped_left
    window = window.reshape(1,86,138,1)
    prob = model.predict(window)
    prob.ravel()
    print(prob)
    print("max ind: ", np.argmax(np.array(prob)))

    
    


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
    print("center: ", centroids[1])

    plt.figure()
    plt.imshow(mask)
    plt.show()

    return top_row, bottom_row, width, centroids[1]


if __name__ == '__main__':
    main()
