import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown
import os
from PIL import Image

# File details
MODEL_FILENAME = "deepfake_resnet50_deepffn.h5"
FILE_ID = "1nmoZPCxnsMcuYVDQvvh8Uwww3OZsZH7q"

# Download the model if not already present
if not os.path.exists(MODEL_FILENAME):
    with st.spinner("Downloading model..."):
        gdown.download(id=FILE_ID, output=MODEL_FILENAME, quiet=False)

# Load the model
@st.cache_resource
def load_deepfake_model():
    model = load_model(MODEL_FILENAME)
    return model

model = load_deepfake_model()

# Streamlit UI
st.set_page_config(page_title="Deepfake Detection", layout="centered")
st.title("🔍 Deepfake Detection App")
st.write("Upload a face image to check if it's **real** or **fake**.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize if needed

    # Make prediction
    prediction = model.predict(img_array)
    label = "Real" if prediction[0][0] < 0.5 else "Fake"
    confidence = (1 - prediction[0][0]) if label == "Real" else prediction[0][0]

    # Display result
    st.markdown(f"### Result: **{label}**")
    st.progress(int(confidence * 100))
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
