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
    "Acne": {"eat": "Paani 3-4 Litre, Coconut Water, Palak, Cucumber, Dal, Oats, Orange, Amla, Dahi", "avoid": "Sugar, Mithai, Cold Drink, Fried, Maida, Fast Food, Chocolate, Chips, Dudh jyada", "remedy": "Chehra din me 2 baar halke paani se dhoyo, Haath se dane mat phodo, Takiya ka cover roz badlo.", "meal": "Subah: Nimbu Paani + Oats | Lunch: Dal + Roti + Salad | Raat: Khichdi + Dahi"},
    "Eczema": {"eat": "Paani jyada, Dahi, Coconut Oil, Palak, Orange, Badam, Oats, Fish", "avoid": "Teekha, Masaledar, Khatta, Pickle, Maida, Bakery, Cold Drink, Dust wali cheez", "remedy": "Nariyal tel lagao, Garam paani se mat nahaao, Cotton kapde pehno, Khujli mat karo.", "meal": "Subah: Doodh + Badam | Lunch: Dal + Roti + Palak | Raat: Daliya + Dahi"},
    "Psoriasis": {"eat": "Paani jyada, Palak, Methi, Haldi, Lahsun, Fish, Dahi, Fruits", "avoid": "Sugar, Maida, Alcohol, Smoking, Fried, Red Meat, Spicy, Bakery", "remedy": "Nariyal tel + Haldi lagao, Dhoop me 10 min baitho, Skin ko sukha mat hone do.", "meal": "Subah: Haldi Doodh + Fruits | Lunch: Dal + Roti + Sabzi | Raat: Khichdi"},
    "Ringworm": {"eat": "Dahi, Haldi, Neem, Lahsun, Green Sabzi, Fruits, Paani 3 Litre", "avoid": "Sugar, Gud, Meetha, Maida, White Bread, Fried, Alcohol", "remedy": "Neem ka Tel + Haldi lagao, Kapde dhoop me sukhhao, Kisi ka towel mat use karo.", "meal": "Subah: Nimbu Paani + Oats | Lunch: Roti + Dal + Sabzi | Raat: Khichdi + Dahi"},
    "Normal Skin": {"eat": "Paani 3 Litre, Fruits, Salad, Dahi, Dal, Badam, Palak, Coconut Water", "avoid": "Jyada Sugar, Jyada Fried, Cold Drink, Junk Food, Smoking", "remedy": "Skin achhi hai! Roz paani piyo, Sunscreen lagao, Neend puri lo.", "meal": "Subah: Fruits + Oats | Lunch: Dal + Roti + Salad | Raat: Halka Khana"},
    "Melanoma": {"eat": "Paani jyada, Palak, Orange, Amla, Haldi, Dahi, Dal, Fruits", "avoid": "Sugar, Maida, Alcohol, Smoking, Fried, Processed Food, Dhoop me jyada rehna", "remedy": "Isme ghar ka upay mat karo! Turant bade Doctor / Hospital jao. Dhoop me mat jao, Daag ko mat chhedo.", "meal": "Subah: Fruits + Doodh | Lunch: Dal + Roti + Salad | Raat: Khichdi - Doctor se diet puchho"},
    "Vitiligo": {"eat": "Palak, Methi, Chukandar, Amla, Orange, Badam, Akhrot, Dahi, Dal, Paani jyada", "avoid": "Khatta - Nimbu, Pickle, Dahi + Machhi ek saath nahi, Maida, Bakery, Alcohol, Smoking", "remedy": "Dhoop me subah 10 min baitho, Stress mat lo, Adrak ka ras safed daag pe lagao (Doctor se puch ke).", "meal": "Subah: Badam + Doodh | Lunch: Dal + Roti + Palak | Raat: Khichdi + Salad"}
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
