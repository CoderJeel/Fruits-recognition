import tensorflow as tf
import numpy as np
import pickle
import os
from cv2 import cv2

model = tf.keras.models.load_model('/home/jeel/Desktop/Fruites_recg/Fruits_reg.h5')

categories = pickle.load(open('/home/jeel/Desktop/Fruites_recg/categories.pickle',mode='rb'))

import sys

def model_predict(path):

  
  img_height = 200
  img_weight = 200
      
  temp_img = cv2.imread(os.path.abspath(path))   
  new_array = cv2.resize(temp_img,(img_weight,img_height))
  img_data_temp = np.array(new_array).reshape(-3,img_height,img_weight,3)
  img_data_temp = img_data_temp/255.0
  pred = model.predict_classes(img_data_temp)

  return pred



path = input('enter path : ')

pred = model_predict(path)

print(pred)




