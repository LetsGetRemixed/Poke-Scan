import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def train_model(cards,labels):
    print("Initiated train_model")

    model = Sequential()

    h,w = cards[0].shape

    model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu', input_shape = (h,w,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    '''model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))'''

    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(205,activation='softmax'))
    model.add(Dropout(0.2))

    '''model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(204, activation='softmax'))'''



    cards = np.divide(cards,255)


    cards = np.array(cards)
    cards = cards.reshape(len(labels),h,w,1)

    cards,labels = shuffle(cards,labels,random_state=0)

    labels = np.array(labels)

    model.compile(optimizer=Adam(learning_rate=1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    BATCH = 128
    EPOCH = 25

    history = model.fit(cards,labels,batch_size=BATCH,epochs=EPOCH)
    
    return model, history