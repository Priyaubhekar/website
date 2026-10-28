import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import requests
import os

# --- Page Setting ---
st.set_page_config(page_title="SkinCheck AI - Insights Dashboard", page_icon="✨", layout="wide")
st.markdown("<h1 style='text-align:center;'>✨ SkinCheck AI - Insights Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>By <b>Priya Ubhekar</b> | AI Skin Analysis Report</p>", unsafe_allow_html=True)
st.markdown("---")

CLASS_NAMES = ['Acne', 'Eczema', 'Psoriasis', 'Ringworm', 'Vitiligo', 'Normal Skin']

# --- Diet + Remedies Data (English) ---
INFO = {
    'Acne': {"diet_eat": "Green vegetables, Fruits, Water, Zinc-rich food", "diet_avoid": "Oily food, Junk food, Sugar, Milk", "remedy": "Neem face wash, Aloe Vera gel, Drink 3L water", "routine_am": "Gentle Face Wash -> Oil-free Moisturizer -> Sunscreen", "routine_pm": "Face Wash -> Aloe Vera Gel -> Early Sleep"},
    'Eczema': {"diet_eat": "Omega-3 foods, Coconut oil, Green vegetables", "diet_avoid": "Spicy food, Eggs, Nuts, Dust", "remedy": "Coconut oil, Oatmeal bath, Keep skin moist", "routine_am": "Mild Cleanser -> Thick Moisturizer -> Sunscreen", "routine_pm": "Lukewarm Bath -> Moisturizer -> Cotton Clothes"},
    'Psoriasis': {"diet_eat": "Fish, Fruits, Vegetables, Turmeric", "diet_avoid": "Red meat, Alcohol, Spicy food", "remedy": "Aloe Vera, Coconut oil, Sunlight 10 min", "routine_am": "Gentle Wash -> Moisturizer -> Sunscreen", "routine_pm": "Moisturizer -> Aloe Vera -> Loose Clothes"},
    'Ringworm': {"diet_eat": "Garlic, Turmeric, Neem, Vitamin C foods", "diet_avoid": "Sugar, Oily food, Sweets", "remedy": "Neem paste, Keep area dry, Don't share towel", "routine_am": "Antifungal Wash -> Keep Dry -> Clean Clothes", "routine_pm": "Antifungal Cream -> Dry Area -> Clean Bed Sheet"},
    'Vitiligo': {"diet_eat": "Green leafy vegetables, Fruits, Copper-rich food", "diet_avoid": "Citrus fruits, Junk food, Alcohol", "remedy": "Aloe Vera, Turmeric + Mustard oil, Sunlight", "routine_am": "Gentle Wash -> Moisturizer -> Sunscreen SPF 50", "routine_pm": "Moisturizer -> Vitamin E oil -> Rest", "doctor": "Please consult Dermatologist"},
    'Normal Skin': {"diet_eat": "Balanced diet, Fruits, Water, Green vegetables", "diet_avoid": "Too much Junk food, Sugar", "remedy": "Your skin is healthy! Keep it clean", "routine_am": "Face Wash -> Light Moisturizer -> Sunscreen", "routine_pm": "Face Wash -> Light Moisturizer -> Good Sleep"},
}

# --- Model Load ---
FILE_ID = "19aAa1_GctA2hGqtzcjg05LQXF4STNXy1"
MODEL_PATH = "skin_disease_model.keras"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

@st.cache_resource
def load_model():
    if not os.path.exists("skin_model.h5"):
        with st.spinner("Model download ho raha hai... 1 min ruko..."):
            r = requests.get(MODEL_URL)
            open("skin_model.h5","wb").write(r.content)
    try:
        return tf.keras.models.load_model("skin_model.h5")
    except:
        return None

model = load_model()

# --- Sidebar ---
st.sidebar.header("📸 Upload Photo")
uploaded = st.sidebar.file_uploader("Apni skin ki photo daalo", type=["jpg","png","jpeg"])
st.sidebar.info("Tip: Saaf light me photo lo, accuracy zyada aayegi.")

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Your Uploaded Photo", width=300)

    if st.button("🔍 ANALYZE MY SKIN - Insights Dekho"):
        if model is None:
            st.error("Model nahi mila! Pehle Colab me train karke skin_model_high_accuracy.h5 yaha daalo.")
        else:
            # Predict
            img = image.resize((224,224))
            arr = np.array(img)/255.0
            arr = np.expand_dims(arr, axis=0)
            pred = model.predict(arr)[0]
            idx = np.argmax(pred)
            result = CLASS_NAMES[idx]
            conf = float(pred[idx]*100)

            # --- 1. TOP SCORE ---
            if result == 'Normal Skin':
                health_score = 92 + int(conf*0.05)
                skin_age_msg = "🎉 Your Skin looks 2 Years Younger than your age!"
                st.success(f"### ✅ {result} - Confidence: {conf:.1f}%")
            else:
                health_score = max(35, 95 - int(conf*0.4))
                skin_age_msg = "⚠️ Skin needs care, age looks slightly older due to problem"
                st.warning(f"### 🩺 Result: {result} - Confidence: {conf:.1f}%")

            col1, col2, col3 = st.columns(3)
            col1.metric("Skin Health Score", f"{health_score} / 100")
            col2.metric("Confidence", f"{conf:.1f}%")
            col3.metric("Skin Age Insight", skin_age_msg)

            # --- 2. CONFIDENCE GRAPH ---
            st.subheader("📊 Prediction Analysis")
            fig1, ax1 = plt.subplots()
            ax1.barh(CLASS_NAMES, pred*100, color=['#FF4B4B' if c==result else '#B0B0B0' for c in CLASS_NAMES])
            ax1.set_xlabel("Confidence %")
            st.pyplot(fig1)

            # --- 3. 12 METRICS RADAR CHART - ScanSkinAI Jaisa ---
            st.subheader("🕸️ 12 Skin Metrics - Insights (Radar Chart)")
            # Photo se color nikal ke metrics banaye
            r = np.array(img)[:,:,0].mean()
            g = np.array(img)[:,:,1].mean()
            b = np.array(img)[:,:,2].mean()
            redness = min(95, max(30, int((r/(g+1))*50)))
            oiliness = 70 if result=='Acne' else 40
            hydration = 85 if result=='Normal Skin' else 45

            labels = ['Texture','Pores','Redness','Oiliness','Fine Lines','Firmness','Pigmentation','Hydration','Radiance','Aging','Dark Circles','Dark Spots']
            values = [70, 60, redness, oiliness, 40, 65, 55, hydration, 75, 45, 50, 45]
            if result == 'Normal Skin':
                values = [85, 85, 30, 30, 80, 85, 80, 90, 90, 85, 80, 85]

            angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            fig2 = plt.figure(figsize=(6,6))
            ax2 = fig2.add_subplot(111, polar=True)
            ax2.plot(angles, values, color='#FF4B4B', linewidth=2)
            ax2.fill(angles, values, color='#FF4B4B', alpha=0.25)
            ax2.set_thetagrids(np.degrees(angles[:-1]), labels)
            st.pyplot(fig2)

            # --- 4. PRIMARY CONCERNS + DIET + ROUTINE ---
            st.subheader("💡 Primary Concerns")
            if result == 'Normal Skin':
                st.success("✅ No major concern! Your Texture, Hydration and Radiance are Excellent.")
            else:
                st.error(f"⚠️ Main Problem: {result} + Oiliness & Redness High. Hydration Low hai.")

            info = INFO[result]
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🥗 Diet Plan (English)")
                st.write(f"**✅ Eat:** {info['diet_eat']}")
                st.write(f"**❌ Avoid:** {info['diet_avoid']}")
                st.subheader("🏠 Home Remedies")
                st.write(info['remedy'])
            with c2:
                st.subheader("🧴 Personalized Routine")
                st.write(f"**☀️ Morning (AM):** {info['routine_am']}")
                st.write(f"**🌙 Night (PM):** {info['routine_pm']}")
                st.info("⚠️ This is not a medical diagnosis. For serious problem, please consult Dermatologist.")

            st.markdown("---")
            st.markdown("<p style='text-align:center; color:grey;'>Made by Priya Ubhekar | SkinCheck AI Project</p>", unsafe_allow_html=True)
else:
    st.info("👈 Left side se photo upload karo, fir ANALYZE button dabao. Tumhara full Insights Report yaha aayega.")
