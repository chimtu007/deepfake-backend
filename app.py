import streamlit as st
import numpy as np
import gdown
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Constants
MODEL_PATH = 'deepfake_model.h5'
FILE_ID = '1nxuez46yUSBXNcnQSTnbRpSsEPI5NNNm'

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner('Downloading model...'):
        gdown.download(f'https://drive.google.com/uc?id={FILE_ID}', MODEL_PATH, quiet=False)

# Load model
@st.cache_resource
def load_deepfake_model():
    return load_model(MODEL_PATH)

model = load_deepfake_model()

# Streamlit UI
st.title("🧠 Deepfake Detector")
st.write("Upload an image to check if it's Real or Fake")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            img = img.resize((224, 224))  # adjust as per your model
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array)[0][0]
            label = "Fake" if prediction > 0.5 else "Real"
            confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)

            st.success(f"Prediction: **{label}** with confidence **{confidence:.2f}**")
