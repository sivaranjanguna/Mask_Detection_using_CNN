import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np  
import matplotlib.image as mpimg


model = load_model("Face_Mask_detection_Model.keras")

st.title("Face Mask Detection")
st.header("Upload an image to check if a person is wearing a face mask or not.")
st.warning("Note: Please upload images in .jpg format only.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_resized = image.resize((128, 128))
    st.image(image_resized, caption='Uploaded Image.',use_container_width=True)
    image_array = np.array(image_resized) / 255.0
    image_reshaped = np.reshape(image_array, (1, 128, 128, 3))
    prediction = model.predict(image_reshaped)
    pred_label = np.argmax(prediction)
    if pred_label == 0:
        st.success("Prediction: Person is wearing a Face Mask.")
    else:
        st.success("Prediction: Person is NOT wearing a Face Mask.")

