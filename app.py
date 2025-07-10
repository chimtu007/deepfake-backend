from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown

app = Flask(__name__)

MODEL_PATH = 'deepfake_model.h5'
GDRIVE_FILE_ID = '1nxuez46yUSBXNcnQSTnbRpSsEPI5NNNm'

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}', MODEL_PATH, quiet=False)

# Load model
model = load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    img_path = os.path.join('static', file.filename)
    file.save(img_path)

    img = image.load_img(img_path, target_size=(224, 224))  # Change size as needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]
    result = 'Fake' if prediction > 0.5 else 'Real'
    return jsonify({'prediction': result, 'score': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
