import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 

st.header('insect_clasification Model')
model = load_model('C:/Users/N I T R O/Desktop/Insect classification/insect_classi.keras')
data_cat = ['Ant',
 'Bee',
 'Beetle',
 'Butterfly',
 'Dragonfly',
 'Fly',
 'Grasshopper',
 'Ladybug',
 'Mosquito']
img_height = 180
img_width = 180
image =st.text_input('Enter Image name','ant.jpg')

image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width=200)
st.write(f"Insect in image is: {data_cat[np.argmax(score)]}")
st.write(f"With accuracy of: {np.max(score)*100:.2f}%")
