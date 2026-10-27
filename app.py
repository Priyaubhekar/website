import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

st.title("Skin Disease Detection")
st.write("Upload an image to predict")

FILE_ID = "19aAa1_GctA2hGqtzjcjQ5LQXf4STNXy1"
MODEL_PATH = "skin_disease_model.keras"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    st.write("Downloading model from Drive... please wait, 1-2 min...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False)
    st.write("Download done!")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Tumhare 8 classes - isi order me
class_names = [
    "BA-cellulitis",
    "BA-impetigo",
    "FU-athlete-foot",
    "FU-nail-fungus",
    "FU-ringworm",
    "PA-cutaneous-larva-migrans",
    "VI-chickenpox",
    "VI-shingles"
]

display_names = {
    "BA-cellulitis": "Cellulitis",
    "BA-impetigo": "Impetigo",
    "FU-athlete-foot": "Athlete's Foot",
    "FU-nail-fungus": "Nail Fungus",
    "FU-ringworm": "Ringworm",
    "PA-cutaneous-larva-migrans": "Cutaneous Larva Migrans",
    "VI-chickenpox": "Chickenpox",
    "VI-shingles": "Shingles"
}

disease_info = {
    "BA-cellulitis": {"symptoms": "Redness, swelling, warmth, pain", "causes": "Bacterial infection, usually by bacteria entering through broken skin"},
    "BA-impetigo": {"symptoms": "Red sores, blisters, honey-colored crusts", "causes": "Contagious bacterial infection"},
    "FU-athlete-foot": {"symptoms": "Itching, scaling, redness between toes", "causes": "Fungal infection, damp feet"},
    "FU-nail-fungus": {"symptoms": "Thick, discolored, brittle nails", "causes": "Fungal infection of nail"},
    "FU-ringworm": {"symptoms": "Ring-shaped red itchy patch", "causes": "Fungal infection, spread by contact"},
    "PA-cutaneous-larva-migrans": {"symptoms": "Winding red itchy lines on skin", "causes": "Parasite larvae from animal feces in soil/sand"},
    "VI-chickenpox": {"symptoms": "Itchy blisters, fever, tiredness", "causes": "Varicella-zoster virus"},
    "VI-shingles": {"symptoms": "Painful rash, blisters in a band on one side", "causes": "Reactivation of chickenpox virus"}
}

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        # Model ke hisab se size - agar 224 nahi hai to 224 ko badal dena
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        index = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        code = class_names[index]
        name = display_names[code]
        info = disease_info[code]

        st.success(f"### Predicted Disease: {name}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.write(f"**Common Symptoms:** {info['symptoms']}")
        st.write(f"**Causes & General Care:** {info['causes']}")
        st.warning("This is an AI-based research prediction, not a medical diagnosis. Please consult a doctor.")
