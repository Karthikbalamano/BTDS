# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 00:33:13 2022

@author: karth
"""

import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumorClassification10Epochs.h5')
Input_Image_Size=64

image=cv2.imread('C:\\Users\\karth\\brain_tumor\\Training\\BTDS1\\pred\\pred5.jpg')

img=Image.fromarray(image)
img=img.resize((Input_Image_Size,Input_Image_Size))
img=np.array(img)

print(img)

input_img=np.expand_dims(img, axis=0)

model(input_img)




