import os
from unittest import result
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
from matplotlib import pyplot as plt
import random
import torchvision
import torch



app = Flask(__name__)


model =load_model('saved_models/brats_3d_100epochs_simple_unet_weighted_dice.hdf5',compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')


def getResult(img,maskimg):
    # image=cv2.imread(img)
    # image = Image.fromarray(image, 'RGB')
    # image = image.resize((64, 64))
    # image=np.array(image)
    # input_img = np.expand_dims(image, axis=0)
    # result=model.predict(input_img)
    # return result
    test_img = np.load(img)
    test_mask = np.load(maskimg)
    test_mask_argmax=np.argmax(test_mask, axis=3)
    test_img_input = np.expand_dims(test_img, axis=0)
    test_prediction = model.predict(test_img_input)
    test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]
    n_slice = 55
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,n_slice,1], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(test_mask_argmax[:,:,n_slice])
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(test_prediction_argmax[:,:, n_slice])
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'static/result.jpg')
    plt.savefig(file_path)
    return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        f = request.files['imgfile']
        f2 = request.files['maskfile']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        file_path2 = os.path.join(
            basepath, 'uploads', secure_filename(f2.filename))
        f.save(file_path)
        f2.save(file_path2)
        print(file_path)
        print(file_path2)
        getResult(file_path,file_path2)
        return None
    return None

if __name__ == '__main__':
    app.run(debug=True)