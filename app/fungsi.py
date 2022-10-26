import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout,LeakyReLU

def make_model():
    model = Sequential()
    # model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 3), padding='same'))
    # model.add(LeakyReLU(0.1))
    # model.add(Conv2D(32, (3, 3), padding='same'))
    # model.add(LeakyReLU(0.1))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(32, (3, 3), padding='same'))
    # model.add(LeakyReLU(0.1))
    # model.add(Conv2D(64, (3, 3), padding='same'))
    # model.add(LeakyReLU(0.1))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(256))
    # model.add(LeakyReLU(0.1))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation='softmax'))

    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=(224, 224, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model
