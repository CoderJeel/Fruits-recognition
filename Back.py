from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf


model = tf.keras.models.load_model('/home/jeel/ML projects/Fruit_rec/Model/model_v15.h5')


def predict(image1):
    image = load_img(image1,color_mode='grayscale',target_size=(85, 65))
    image = img_to_array(image)
    image = image.reshape((1,85,64,-1))
    image = image/255.0
    pred = model.predict_classes(image)
    return pred

