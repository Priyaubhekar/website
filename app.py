import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

FILE_ID = "19aAa1_GctA2hGqtzcjg05LQXF4STNXy1"
MODEL_PATH = "skin_disease_model.keras"

# Model download - sirf ek baar hoga
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    st.write("Downloading model... please wait 1-2 min...")
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    st.write("Download done!")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

CLASS_NAMES = ['BA- cellulitis', 'BA-impetigo', 'FU-athlete-foot', 'FU-nail-fungus', 'FU-ringworm', 'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles']
DIET_DATA = {
    "BA- cellulitis": {"eat": "Water 3 Litres, Spinach, Turmeric Milk, Orange, Amla, Dal, Curd", "avoid": "Sugar, Sweets, Fried Food, Maida, Alcohol, Junk Food", "remedy": "Keep leg raised, Do not press the swelling, Go to Doctor immediately. This is bacterial infection.", "meal": "Morning: Turmeric Milk + Fruits | Lunch: Dal + Roti + Salad | Night: Khichdi"},
    "BA-impetigo": {"eat": "Water, Turmeric, Neem, Curd, Spinach, Fruits, Dal", "avoid": "Sugar, Sweets, Spicy, Fried Food, Maida", "remedy": "Do not scratch, Do not share towel, Keep skin clean and dry, Apply Neem Oil.", "meal": "Morning: Lemon Water + Oats | Lunch: Dal + Roti + Vegetables | Night: Khichdi + Curd"},
    "FU-athlete-foot": {"eat": "Curd, Garlic, Turmeric, Green Vegetables, Water 3 Litres", "avoid": "Sugar, Jaggery, Maida, White Bread, Alcohol, Fried Food", "remedy": "Keep feet dry, Change socks daily, Dry shoes in sunlight, Do not wear wet shoes.", "meal": "Morning: Lemon Water + Fruits | Lunch: Roti + Dal + Salad | Night: Khichdi"},
    "FU-nail-fungus": {"eat": "Curd, Garlic, Turmeric, Neem, Green Vegetables, Fruits, Water", "avoid": "Sugar, Jaggery, Sweets, Maida, White Bread, Alcohol", "remedy": "Keep nails short and dry, Do not apply nail polish, Soak feet in warm water + salt.", "meal": "Morning: Lemon Water + Oats | Lunch: Dal + Roti + Vegetables | Night: Khichdi + Curd"},
    "FU-ringworm": {"eat": "Curd, Turmeric, Neem, Garlic, Green Vegetables, Fruits, Water 3 Litres", "avoid": "Sugar, Jaggery, Sweets, Maida, White Bread, Fried Food, Alcohol", "remedy": "Apply Neem Oil + Turmeric, Dry clothes in sunlight, Do not use others towel.", "meal": "Morning: Lemon Water + Oats | Lunch: Roti + Dal + Vegetables | Night: Khichdi + Curd"},
    "PA-cutaneous-larva-migrans": {"eat": "Water, Turmeric, Garlic, Spinach, Dal, Fruits, Curd", "avoid": "Sugar, Fried Food, Maida, Alcohol, Junk Food", "remedy": "Do not walk barefoot, Do not scratch, Go to Doctor immediately. This is parasite infection.", "meal": "Morning: Turmeric Milk + Fruits | Lunch: Dal + Roti + Salad | Night: Khichdi"},
    "VI-chickenpox": {"eat": "More Water, Coconut Water, Oats, Daliya, Curd, Fruits, Soft Food", "avoid": "Spicy, Masala, Sour Food, Fried Food, Maida, Cold Drink", "remedy": "Take full rest, Do not scratch blisters, Bath with Neem water, Stay away from others.", "meal": "Morning: Coconut Water + Oats | Lunch: Khichdi + Curd | Night: Daliya"},
    "VI-shingles": {"eat": "Water, Spinach, Orange, Amla, Turmeric, Dal, Curd, Almonds", "avoid": "Sugar, Maida, Alcohol, Smoking, Fried Food, Spicy Food", "remedy": "Take rest, Do not scratch, Keep blisters clean, Go to Doctor. Do not apply cream without Doctor.", "meal": "Morning: Fruits + Milk | Lunch: Dal + Roti + Salad | Night: Khichdi"}
}
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

st.markdown("""<h3 style="color:black;">📸 Step 1: Take Photo</h3>""", unsafe_allow_html=True)
st.markdown("""<p style="color:black;">Choose from gallery OR take a live photo</p>""", unsafe_allow_html=True)

option = st.radio("How do you want to take photo?", ["📁 Upload from Gallery", "📷 Take Live Photo"], horizontal=True)

image = None
uploaded_file = None
camera_photo = None

if option == "📁 Upload from Gallery":
    uploaded_file = st.file_uploader("Upload your skin photo", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
else:
    camera_photo = st.camera_input("📷 Take Live Photo - Keep your skin in front of camera")
    if camera_photo is not None:
        image = Image.open(camera_photo).convert("RGB")

if image is not None:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(image, caption="Your Uploaded Image", use_container_width=True)

    st.markdown("""<h3 style="color:black;">🔍 Step 2: Check Result</h3>""", unsafe_allow_html=True)
    if st.button("🔍 PREDICT NOW"):
        with st.spinner("AI image ko check kar raha hai..."):
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
    if predicted_class in DIET_DATA:
            diet = DIET_DATA[predicted_class]
            st.markdown("---")
            st.markdown(f"""<h3 style="color:black;">🥗 Diet Plan for {predicted_class}</h3>""", unsafe_allow_html=True)

            st.markdown(f"""<div style="background-color:#d4edda; padding:15px; border-radius:10px; color:black; font-size:16px;"><b>✅ EAT:</b> {diet['eat']}</div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""<div style="background-color:#f8d7da; padding:15px; border-radius:10px; color:black; font-size:16px;"><b>❌ AVOID:</b> {diet['avoid']}</div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""<div style="background-color:#cce5ff; padding:15px; border-radius:10px; color:black; font-size:16px;"><b>🏠 Home Remedy:</b> {diet['remedy']}</div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""<div style="background-color:#fff3cd; padding:15px; border-radius:10px; color:black; font-size:16px;"><b>📅 1 Day Meal Plan:</b> {diet['meal']}</div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""<div style="background-color:#fff3cd; padding:15px; border-radius:10px; color:black; font-size:16px;"><b>⚠️ Disclaimer:</b> This is for educational purpose only. Please consult a Doctor.</div>""", unsafe_allow_html=True)
else:
    st.info("👆 Upar se ek skin photo upload karo, fir Predict dabao")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#999; font-size:12px;'>Made for Final Year Project | SkinCheck AI</p>", unsafe_allow_html=True)
