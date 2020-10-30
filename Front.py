import streamlit as st 
from PIL import Image
from Back import predict
import io


st.title("Upload + Classification Example")

uploaded_file = st.file_uploader("Choose an image...", type="jpeg")
if uploaded_file is not None:
    #image = Image.open(io.BytesIO(uploaded_file))
    Image = uploaded_file.read()
    print(type(Image))
    st.image(Image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(Image)
    st.write('%s (%.2f%%)' % (label[1], label[2]*100))