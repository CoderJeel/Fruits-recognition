import tensorflow as tf
import numpy as np
import pickle
import os
from cv2 import cv2
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

model = tf.keras.models.load_model('/home/jeel/ML projects/Fruit_rec/Model/model_v15.h5')

categories = pickle.load(open('/home/jeel/ML projects/Fruit_rec/cat.pickle',mode='rb'))

categories = np.array(list(categories.items()))

import sys

def predict(image1):
    image = load_img(image1,color_mode='grayscale',target_size=(80,65))
    image = img_to_array(image)
    image = np.array(image).reshape(-1,80,65,1)
    image = image/255.0
    pred = model.predict_classes(image)
    return categories[pred]

path = input('Enter path of image:')

print(predict(path))




