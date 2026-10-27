import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import glob
import os

st.title("Skin Disease Detection")
st.write("Upload an image to predict")

# Find model file automatically - no need to type name
model_files = glob.glob("*.h5") + glob.glob("*.keras")
if len(model_files) == 0:
    st.error("Model file not found! Please upload your .h5 or .keras file")
    st.stop()

MODEL_NAME = model_files[0]
st.write(f"Using model: {MODEL_NAME}")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_NAME)

model = load_model()
st.success("Model loaded!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    st.write("Prediction:", prediction)
    st.success(f"Predicted Class: {np.argmax(prediction)}")
