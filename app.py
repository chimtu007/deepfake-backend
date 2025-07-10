import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown
import os
from PIL import Image

# === Constants ===
MODEL_FILENAME = "deepfake_resnet50_deepffn.h5"
GOOGLE_DRIVE_ID = "1nmoZPCxnsMcuYVDQvvh8Uwww3OZsZH7q"

# === Download model from Google Drive if not exists ===
if not os.path.exists(MODEL_FILENAME):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(id=GOOGLE_DRIVE_ID, output=MODEL_FILENAME, quiet=False)

# === Load model (compile=False avoids loading training config) ===
@st.cache_resource
def load_deepfake_model():
    model = load_model(MODEL_FILENAME, compile=False)
    return model

model = load_deepfake_model()

# === Streamlit UI ===
st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("🤖 Deepfake Detection App")
st.markdown("Upload a face image to detect if it's **real** or **deepfake**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # === Preprocess the image ===
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension

    # === Predict ===
    try:
        prediction = model.predict(img_array)
        score = prediction[0][0]
        label = "Real" if score < 0.5 else "Fake"
        confidence = 1 - score if label == "Real" else score

        st.markdown(f"### 🎯 Prediction: **{label}**")
        st.progress(int(confidence * 100))
        st.write(f"Confidence: **{confidence * 100:.2f}%**")

    except Exception as e:
        st.error("❌ Error during prediction.")
        st.exception(e)
