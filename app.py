import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

MODEL_FILENAME = "deepfake_resnet50_deepffn.h5"
DRIVE_FILE_ID = "1nmoZPCxnsMcuYVDQvvh8Uwww3OZsZH7q"  # Extracted from your Drive link

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("Downloading model..."):
            gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_FILENAME, quiet=False)
    return tf.keras.models.load_model(MODEL_FILENAME)

model = load_model()

st.title("Deepfake Detection App")
st.write("Upload a face image to detect whether it's **Real or Fake**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    prediction = model.predict(img_array)[0][0]
    label = "Fake" if prediction > 0.5 else "Real"
    st.markdown(f"### Prediction: **{label}** ({prediction:.2f})")
