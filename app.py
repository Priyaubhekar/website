import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import glob
import os
import requests

st.title("Skin Disease Detection")
st.write("Upload an image to predict")

# --- Download model from Google Drive ---
FILE_ID = "19aAa1_GctA2hGqtzcjg05LQXF4STNXy1"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
MODEL_PATH = "skin_disease_model.keras"

if not os.path.exists(MODEL_PATH):
    st.write("Downloading model from Drive... please wait, 1 min...")
    r = requests.get(MODEL_URL, allow_redirects=True)
    open(MODEL_PATH, 'wb').write(r.content)
    st.write("Download done!")

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
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    st.success(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
