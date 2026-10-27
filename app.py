import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="SkinSense AI", page_icon="🩺", layout="centered")

# --- HIDE BORING STUFF ---
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stApp {background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);}
.upload-box {background: white; padding: 20px; border-radius: 20px; color: black;}
h1, h2, h3, p {color: white !important;}
</style>
""", unsafe_allow_html=True)

FILE_ID = "19aAa1_GctA2hGqtzcjg05LQXF4STNXy1"
MODEL_PATH = "skin_disease_model.keras"

# --- YOUR CLASSES - Apne hisab se naam check kar lena ---
CLASS_NAMES = ['Acne', 'Athletes Foot', 'Eczema', 'Nail Fungus', 'Ringworm', 'Psoriasis', 'Melanoma', 'Vitiligo']

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=True)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --- HEADER ---
st.markdown("<h1 style='text-align:center;'>🩺 SkinSense AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; opacity:0.8;'>Upload your skin image & get instant AI analysis</p>", unsafe_allow_html=True)

# --- UPLOAD ---
uploaded = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"], label_visibility="collapsed")

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    
    col1, col2 = st.columns([1,1.2])
    with col1:
        st.image(image, caption="Your Image", use_column_width=True)
    with col2:
        st.markdown("### 🔍 Analysis")
        with st.spinner("AI is checking..."):
            img = image.resize((224,224))
            arr = np.array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            pred = model.predict(arr, verbose=0)[0]
            
            top_idx = np.argmax(pred)
            top_conf = pred[top_idx]*100
            
            # Result Card
            if top_conf > 70:
                st.success(f"**Predicted: {CLASS_NAMES[top_idx]}**")
            elif top_conf > 45:
                st.warning(f"**Maybe: {CLASS_NAMES[top_idx]}**")
            else:
                st.error(f"**Not Sure: {CLASS_NAMES[top_idx]}** - Photo is unclear")
            
            st.markdown(f"**Confidence: {top_conf:.1f}%**")
            st.progress(float(pred[top_idx]))

            st.markdown("---")
            st.markdown("**Top 3 Possibilities:**")
            top3 = np.argsort(pred)[::-1][:3]
            for i in top3:
                st.write(f"{CLASS_NAMES[i]} - {pred[i]*100:.1f}%")
                st.progress(float(pred[i]))
    
    st.info("💡 **Tip:** For better result, take photo in daylight, close-up, without flash. This is for project purpose only, please consult a doctor.")
else:
    st.markdown("""
    <div style='background:white; padding:25px; border-radius:20px; text-align:center; color:black;'>
    <h3 style='color:black !important;'>📸 How to use?</h3>
    <p style='color:#475569 !important;'>1. Click Browse / Upload<br>2. Select clear skin photo<br>3. Get result in 2 seconds</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<p style='text-align:center; font-size:12px; opacity:0.5; margin-top:30px;'>Made with ❤️ for Final Year Project</p>", unsafe_allow_html=True)
