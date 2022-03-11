# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 00:20:51 2022

@author: karth
"""

import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical

Tain_path = 'dataset/'

without_tumor_images = os.listdir(Tain_path+ 'no/')
with_tumor_images = os.listdir(Tain_path+ 'yes/')
dataset=[]
label=[]
Img_Input_Size=64

# print(no_tumor_images)

for i, image_name in enumerate(without_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(Tain_path+'no/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((Img_Input_Size,Img_Input_Size))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(with_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(Tain_path+'yes/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((Img_Input_Size,Img_Input_Size))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)
x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2)

# Reshape = (n, image_width, image_height, n_channel)

# print(x_train.shape)
# print(y_train.shape)

# print(x_test.shape)
# print(y_test.shape)

x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)


# Model Building
# 64,64,3

model=Sequential()
model.add(Conv2D(32, (3,3), input_shape=(Img_Input_Size, Img_Input_Size, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test),shuffle=False)
model.save('BrainTumorClassification10Epochs.h5')




