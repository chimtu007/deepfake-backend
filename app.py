from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import gdown

app = Flask(__name__)
CORS(app)

# MODEL FILE SETUP
model_file = "deepfake_resnet50_deepffn.h5"
model_url = "https://drive.google.com/uc?export=download&id=1nmoZPCxnsMcuYVDQvvh8Uwww3OZsZH7q"

# Download model if not already present
if not os.path.exists(model_file):
    print("Downloading model...")
    gdown.download(model_url, model_file, quiet=False)

# Load the model
model = load_model(model_file)

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files["file"]
    img = Image.open(file.stream)
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)[0][0]
    result = "FAKE" if prediction >= 0.5 else "REAL"
    confidence = round(prediction if result == "FAKE" else 1 - prediction, 2)
    return jsonify({"result": result, "confidence": confidence * 100})

if __name__ == "__main__":
    app.run(debug=True)
