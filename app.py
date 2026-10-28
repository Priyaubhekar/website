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

CLASS_NAMES = ['Acne', 'Eczema', 'Psoriasis', 'Ringworm', 'Normal Skin', 'Melanoma', 'Vitiligo']
DIET_DATA = {
    "Acne": {"eat": "Water 3-4 Litres, Coconut Water, Spinach, Cucumber, Lentils (Dal), Oats, Orange, Amla, Curd", "avoid": "Sugar, Sweets, Cold Drink, Fried Food, Maida, Fast Food, Chocolate, Chips, Excess Milk", "remedy": "Wash face 2 times a day with plain water, Do not pop pimples, Change pillow cover daily.", "meal": "Morning: Lemon Water + Oats | Lunch: Dal + Roti + Salad | Night: Khichdi + Curd"},
    "Eczema": {"eat": "More Water, Curd, Coconut Oil, Spinach, Orange, Almonds, Oats, Fish", "avoid": "Spicy, Masala, Sour Food, Pickle, Maida, Bakery, Cold Drink, Dust", "remedy": "Apply Coconut Oil, Do not bathe with hot water, Wear cotton clothes, Do not scratch.", "meal": "Morning: Milk + Almonds | Lunch: Dal + Roti + Spinach | Night: Daliya + Curd"},
    "Psoriasis": {"eat": "More Water, Spinach, Fenugreek (Methi), Turmeric, Garlic, Fish, Curd, Fruits", "avoid": "Sugar, Maida, Alcohol, Smoking, Fried Food, Red Meat, Spicy Food, Bakery", "remedy": "Apply Coconut Oil + Turmeric, Sit in sunlight for 10 mins, Do not let skin become dry.", "meal": "Morning: Turmeric Milk + Fruits | Lunch: Dal + Roti + Vegetables | Night: Khichdi"},
    "Ringworm": {"eat": "Curd, Turmeric, Neem, Garlic, Green Vegetables, Fruits, Water 3 Litres", "avoid": "Sugar, Jaggery, Sweets, Maida, White Bread, Fried Food, Alcohol", "remedy": "Apply Neem Oil + Turmeric, Dry clothes in sunlight, Do not use others towel.", "meal": "Morning: Lemon Water + Oats | Lunch: Roti + Dal + Vegetables | Night: Khichdi + Curd"},
    "Normal Skin": {"eat": "Water 3 Litres, Fruits, Salad, Curd, Dal, Almonds, Spinach, Coconut Water", "avoid": "Too Much Sugar, Too Much Fried Food, Cold Drink, Junk Food, Smoking", "remedy": "Your skin is healthy! Drink water daily, Apply sunscreen, Take full sleep.", "meal": "Morning: Fruits + Oats | Lunch: Dal + Roti + Salad | Night: Light Dinner"},
    "Melanoma": {"eat": "More Water, Spinach, Orange, Amla, Turmeric, Curd, Dal, Fruits", "avoid": "Sugar, Maida, Alcohol, Smoking, Fried Food, Processed Food, Too much Sun exposure", "remedy": "Do not try home remedy! Go to a big Doctor / Hospital immediately. Do not go in sun, Do not touch the spot.", "meal": "Morning: Fruits + Milk | Lunch: Dal + Roti + Salad | Night: Khichdi - Ask Doctor for diet"},
    "Vitiligo": {"eat": "Spinach, Fenugreek, Beetroot, Amla, Orange, Almonds, Walnuts, Curd, Dal, More Water", "avoid": "Sour Food - Lemon, Pickle, Curd + Fish together, Maida, Bakery, Alcohol, Smoking", "remedy": "Sit in morning sunlight for 10 mins, Do not take stress, Apply ginger juice on white patches (after asking Doctor).", "meal": "Morning: Almonds + Milk | Lunch: Dal + Roti + Spinach | Night: Khichdi + Salad"}
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
                st.markdown(f"### 🥗 {predicted_class} ke liye Diet Plan")
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**✅ KHAO:** {diet['eat']}")
                with col2:
                    st.error(f"**❌ MAT KHAO:** {diet['avoid']}")
                st.info(f"**🏠 Ghar ka Upay:** {diet['remedy']}")
                st.warning(f"**📅 1 Din ka Meal:** {diet['meal']}")
        st.warning("⚠️ Disclaimer: Ye sirf educational purpose ke liye hai. Confirm karne ke liye Dermatologist se consult karein.")
else:
    st.info("👆 Upar se ek skin photo upload karo, fir Predict dabao")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#999; font-size:12px;'>Made for Final Year Project | SkinCheck AI</p>", unsafe_allow_html=True)
