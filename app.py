import streamlit as st
import numpy as np
import gdown
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- Constants ---
MODEL_PATH = "deepfake_cnn_model.h5"
GDRIVE_FILE_ID = "1nxuez46yUSBXNcnQSTnbRpSsEPI5NNNm"  # your new .h5 file link ID

# --- Download model if needed ---
if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model..."):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# --- Load model ---
@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)

model = load_model_cached()

# --- Streamlit UI ---
st.title("🧠 Deepfake CNN Detector")
st.write("Upload a face image to predict if it's **Real** or **Fake**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            img = img.resize((128, 128))  # match training size
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0][0]
            label = "Fake" if prediction > 0.5 else "Real"
            confidence = prediction if prediction > 0.5 else 1 - prediction

            st.success(f"Prediction: **{label}** with confidence **{confidence:.2f}**")
