# created by Kai Hoshijo
# date: 3/26/19
# time: 11:40 PM
# mood: tired

import keras
import numpy as np
from keras import layers

#%%
import generateData

#%%
if __name__ == "__main__": 

    generate = generateData.generateData(400)


    x_train, x_test, y_train, y_test = generate.splitData(0.33, 33)
    # print(x_train.head())zzzz
    # print(y_train.head()) 
    img_shape = generate.x[0].shape[1:]

    model = keras.Sequential()
    # convn layer
    model.add(layers.Conv2D(32, kernel_size = (3,3), activation = "relu"))
    input_shape = (img_shape, img_shape, 1)
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.BatchNormalization())
    # conv layer
    model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.BatchNormalization())
    # conv layer
    model.add(layers.Conv2D(96, kernel_size=(4,4), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.BatchNormalization())
    # conv layer
    model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.BatchNormalization())
    # conv layer
    model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.7))

    # connected layer
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(4, activation = 'softmax'))

    optimizer = keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, decay = 0.0, amsgrad = False)
    # optimizer = keras.optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    # optimizer = keras.optimizers.Adadelta()
    model.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])


    model.fit(x_train, y_train, batch_size = 50, epochs = 80, verbose = 1)

    print(model.evaluate(x_test, y_test))
