import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown
import os
from PIL import Image

# === Download model if not present ===
MODEL_FILENAME = "deepfake_resnet50_deepffn.h5"
FILE_ID = "1nmoZPCxnsMcuYVDQvvh8Uwww3OZsZH7q"

if not os.path.exists(MODEL_FILENAME):
    with st.spinner("Downloading model..."):
        gdown.download(id=FILE_ID, output=MODEL_FILENAME, quiet=False)

# === Load model ===
@st.cache_resource
def load_deepfake_model():
    model = load_model(MODEL_FILENAME)
    return model

model = load_deepfake_model()

# === Streamlit UI ===
st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("🤖 Deepfake Detector with ResNet50 + DFFNN")
st.write("Upload a face image (jpg/png). Model will classify it as Real or Fake.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # === Preprocess to (224, 224, 3) and normalize ===
    img = img.convert("RGB")  # Ensure 3 channels
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)

    img_array = img_array / 255.0  # Normalize if required by your training
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

    # === Predict ===
    try:
        prediction = model.predict(img_array)
        label = "Real" if prediction[0][0] < 0.5 else "Fake"
        confidence = 1 - prediction[0][0] if label == "Real" else prediction[0][0]

        st.markdown(f"### 🔍 Prediction: **{label}**")
        st.progress(int(confidence * 100))
        st.write(f"Confidence: **{confidence * 100:.2f}%**")

    except Exception as e:
        st.error("⚠️ Error during prediction.")
        st.exception(e)
