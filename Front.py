import streamlit as st 
from PIL import Image
from Back import predict
import io


st.title("Fruit Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    print(type(image))
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = test(uploaded_file)
    st.write('%s (%.2f%%)' % (label[1], label[2]*100))