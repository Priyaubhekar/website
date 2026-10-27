import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

st.title("Skin Disease Detection")
st.write("Upload an image to predict")

FILE_ID = "19aAa1_GctA2hGqtzcjg05LQXF4STNXy1"
MODEL_PATH = "skin_disease_model.keras"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    st.write("Downloading model from Drive... please wait, 1-2 min...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
  gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
    st.write("Download done!")
    st.write(f"File size: {os.path.getsize(MODEL_PATH)} bytes")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
st.success("Model loaded!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    st.success(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
