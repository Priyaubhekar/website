import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

FILE_ID = "1Tor1YFYlqWwY5dhZ0BVdWcSpYrN5-cKF"
MODEL_PATH = "new_model_9class.keras"

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

CLASS_NAMES = ['BA- cellulitis', 'BA-impetigo', 'FU-athlete-foot', 'FU-nail-fungus', 'FU-ringworm', 'Healthy', 'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles']

DIET_DATA = {
    "BA- cellulitis": {"eat": "Water 3 Litres, Spinach, Turmeric Milk, Orange, Amla, Dal, Curd", "avoid": "Sugar, Sweets, Fried Food, Maida, Alcohol, Junk Food", "remedy": "Keep leg raised, Do not press the swelling, Go to Doctor immediately. This is bacterial infection.", "meal": "Morning: Turmeric Milk + Fruits | Lunch: Dal + Roti + Salad | Night: Khichdi"},
    "BA-impetigo": {"eat": "Water, Turmeric, Neem, Curd, Spinach, Fruits, Dal", "avoid": "Sugar, Sweets, Spicy, Fried Food, Maida", "remedy": "Do not scratch, Do not share towel, Keep skin clean and dry, Apply Neem Oil.", "meal": "Morning: Lemon Water + Oats | Lunch: Dal + Roti + Vegetables | Night: Khichdi + Curd"},
    "FU-athlete-foot": {"eat": "Curd, Garlic, Turmeric, Green Vegetables, Water 3 Litres", "avoid": "Sugar, Jaggery, Maida, White Bread, Alcohol, Fried Food", "remedy": "Keep feet dry, Change socks daily, Dry shoes in sunlight, Do not wear wet shoes.", "meal": "Morning: Lemon Water + Fruits | Lunch: Roti + Dal + Salad | Night: Khichdi"},
    "FU-nail-fungus": {"eat": "Curd, Garlic, Turmeric, Neem, Green Vegetables, Fruits, Water", "avoid": "Sugar, Jaggery, Sweets, Maida, White Bread, Alcohol", "remedy": "Keep nails short and dry, Do not apply nail polish, Soak feet in warm water + salt.", "meal": "Morning: Lemon Water + Oats | Lunch: Dal + Roti + Vegetables | Night: Khichdi + Curd"},
    "FU-ringworm": {"eat": "Curd, Turmeric, Neem, Garlic, Green Vegetables, Fruits, Water 3 Litres", "avoid": "Sugar, Jaggery, Sweets, Maida, White Bread, Fried Food, Alcohol", "remedy": "Apply Neem Oil + Turmeric, Dry clothes in sunlight, Do not use others towel.", "meal": "Morning: Lemon Water + Oats | Lunch: Roti + Dal + Vegetables | Night: Khichdi + Curd"},
    "Healthy": {"eat": "Balanced Diet, Green Vegetables, Fruits, Dal, Curd, Water 3 Litres, Dry Fruits", "avoid": "Junk Food, Too Much Sugar, Fried Food, Cold Drink, Alcohol", "remedy": "Skin is healthy! Keep it clean, Drink water, Use Sunscreen, Eat healthy food daily.", "meal": "Morning: Fruits + Milk / Oats | Lunch: Dal + Roti + Salad + Curd | Night: Light Khichdi / Vegetables"},
    "PA-cutaneous-larva-migrans": {"eat": "Water, Turmeric, Garlic, Spinach, Dal, Fruits, Curd", "avoid": "Sugar, Fried Food, Maida, Alcohol, Junk Food", "remedy": "Do not walk barefoot, Do not scratch, Go to Doctor immediately. This is parasite infection.", "meal": "Morning: Turmeric Milk + Fruits | Lunch: Dal + Roti + Salad | Night: Khichdi"},
    "VI-chickenpox": {"eat": "More Water, Coconut Water, Oats, Daliya, Curd, Fruits, Soft Food", "avoid": "Spicy, Masala, Sour Food, Fried Food, Maida, Cold Drink", "remedy": "Take full rest, Do not scratch blisters, Bath with Neem water, Stay away from others.", "meal": "Morning: Coconut Water + Oats | Lunch: Khichdi + Curd | Night: Daliya"},
    "VI-shingles": {"eat": "Water, Spinach, Orange, Amla, Turmeric, Dal, Curd, Almonds", "avoid": "Sugar, Maida, Alcohol, Smoking, Fried Food, Spicy Food", "remedy": "Take rest, Do not scratch, Keep blisters clean, Go to Doctor. Do not apply cream without Doctor.", "meal": "Morning: Fruits + Milk | Lunch: Dal + Roti + Salad | Night: Khichdi"}
}

st.set_page_config(page_title="SkinCheck AI", page_icon="🔍", layout="centered")
st.markdown("""<style>
    .stApp { background-color: #F0F4FF; }
    .main-title { text-align: center; color: #1E2A78; font-size: 38px; font-weight: 800; margin-bottom: 0px; line-height: 1.1; }
    .sub-title { text-align: center; color: #666; font-size: 16px; margin-top: 5px; }
    .header-card {
        background: linear-gradient(135deg, #4A6CF7, #8A4DFF);
        padding: 28px 20px 22px 20px;
        border-radius: 22px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(74,108,247,0.30);
        margin-bottom: 18px;
    }
    .header-card h1 { color: white !important; font-size: 34px; margin: 0px; }
    .header-card p { color: #E0E7FF !important; font-size: 15px; margin: 8px 0 0 0; }
    .stButton>button {
        background: linear-gradient(90deg, #4A6CF7, #8A4DFF);
        color: white !important;
        border: none;
        padding: 13px 30px;
        border-radius: 25px;
        font-size: 17px;
        font-weight: bold;
        width: 100%;
        box-shadow: 0 4px 15px rgba(74,108,247,0.35);
    }
    .result-card {
        background: white;
        padding: 22px;
        border-radius: 20px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        border: 1px solid #EDEFFF;
    }
    [data-testid="stFileUploader"], [data-testid="stCameraInput"] {
        background: white;
        padding: 12px;
        border-radius: 16px;
        border: 1px solid #EDEFFF;
    }
    .stRadio>div { background: white; padding: 10px 12px; border-radius: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-card">
    <h1>🧴 SkinCheck AI</h1>
    <p>Upload Photo • Instant AI Analysis • Diet + Medicine Info</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""<p style="text-align:center; color:#666; font-size:13px; margin-top:-8px;">🔒 100% Private • No photo is saved • Made for Final Year Project</p>""", unsafe_allow_html=True)
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
if st.button("✨ PREDICT NOW"):
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
                st.markdown(f"""<div style="background-color:#D4EDDA; padding:15px; border-radius:12px; color:black; margin-bottom:10px;"><b>✅ Eat This:</b><br>{diet['eat']}</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style="background-color:#F8D7DA; padding:15px; border-radius:12px; color:black; margin-bottom:10px;"><b>🚫 Avoid This:</b><br>{diet['avoid']}</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style="background-color:#FFF3CD; padding:15px; border-radius:12px; color:black; margin-bottom:10px;"><b>🌿 Home Remedy:</b><br>{diet['remedy']}</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style="background-color:#CCE5FF; padding:15px; border-radius:12px; color:black;"><b>🍽️ Meal Plan:</b><br>{diet['meal']}</div>""", unsafe_allow_html=True)
else:
    st.info("👆 Please upload a skin photo from above, then press Predict")
