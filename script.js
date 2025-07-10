// Wait for the page to load
window.addEventListener('load', () => {
  const upload = document.getElementById('upload');
  const resultText = document.getElementById('result');

  let model;

  // Load the TensorFlow.js model
  async function loadModel() {
    resultText.innerText = 'Loading model...';
    model = await tf.loadLayersModel('model.json');
    resultText.innerText = 'Model loaded. Upload an image.';
  }

  // Preprocess image
  function preprocessImage(imageElement) {
    return tf.tidy(() => {
      let tensor = tf.browser.fromPixels(imageElement)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(255.0)
        .expandDims();
      return tensor;
    });
  }

  // Predict and show result
  async function predict(imageElement) {
    const tensor = preprocessImage(imageElement);
    const prediction = await model.predict(tensor).data();
    const result = prediction[0] > 0.5 ? '⚠️ Fake' : '✅ Real';
    resultText.innerText = `Prediction: ${result} (${prediction[0].toFixed(4)})`;
  }

  // Handle image upload
  upload.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
      await predict(img);
    };
  });

  loadModel(); // Load model on page load
});
