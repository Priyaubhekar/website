import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# ===== 1. MODEL PART - PURANA WALA, ISKO MAT CHEDNA =====
FILE_ID = "1BaAl_GtA2hMqtzcJg5LQ4f45T8Ny1"
MODEL_PATH = "skin_disease_model.keras"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    st.write("Downloading model... please wait 1-2 min...")
    gdown.download(FILE_ID, output=MODEL_PATH, quiet=False)
    st.write("Download done!")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Tumhare purane wale class names - same order rakhna
CLASS_NAMES = ['Acne', 'Eczema', 'Psoriasis', 'Ringworm', 'Normal Skin', 'Melanoma', 'Vitiligo']

# ===== 2. NEW INTERFACE - SIRF YAHAN CHANGE KIYA HAI =====
st.set_page_config(page_title="SkinCheck AI", page_icon="🩺", layout="centered")

st.markdown("""
<style>
.stApp { background-color: #F0F4FF; }
.main-title { text-align: center; color: #1E2A78; font-size: 38px; font-weight: 800; margin-bottom: 0px; }
.sub-title { text-align: center; color: #666; font-size: 16px; margin-top: 5px; }
.stButton>button { background: linear-gradient(90deg, #4A6CF7, #8A4DFF); color: white; border-radius: 12px; padding: 12px; font-size: 18px; font-weight: bold; border: none; width: 100%; }
.result-card { background: white; padding: 25px; border-radius: 18px; box-shadow: 0px 4px 20px rgba(0,0,0,0.08); text-align: center; border-top: 5px solid #4A6CF7; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🩺 SkinCheck AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI Powered Skin Disease Detection</p>', unsafe_allow_html=True)
st.markdown("---")

st.markdown("### 📸 Step 1: Upload Skin Photo")
uploaded_file = st.file_uploader("JPG / PNG photo upload karo", type=["jpg","jpeg","png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(image, caption="Your Uploaded Image", use_container_width=True)

    st.markdown("### 🤖 Step 2: Check Result")
    if st.button("🔍 PREDICT NOW"):
        with st.spinner("AI image ko check kar raha hai..."):
            # SAME PREPROCESSING - ISKO MAT BADALNA
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

        st.markdown(f"""
        <div class="result-card">
            <p style="color:grey; margin:0;">Prediction Result</p>
            <h1 style="color:#4A6CF7; margin:5px;">{predicted_class}</h1>
            <p style="font-size:18px;">Confidence: <b>{confidence:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(confidence))
        st.warning("⚠️ Disclaimer: Ye sirf educational purpose ke liye hai. Confirm karne ke liye Dermatologist se consult karein.")
else:
    st.info("👆 Upar se ek skin photo upload karo, fir Predict dabao")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#999; font-size:12px;'>Made for Final Year Project | SkinCheck AI</p>", unsafe_allow_html=True)
