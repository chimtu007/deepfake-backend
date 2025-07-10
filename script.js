let model;
const resultEl = document.getElementById("result");

async function loadModel() {
  try {
    resultEl.textContent = "Loading model...";
    model = await tf.loadLayersModel("model.json");
    resultEl.textContent = "Model loaded. Upload an image.";
  } catch (error) {
    console.error("Model load error:", error);
    resultEl.textContent = "Failed to load model.";
  }
}

loadModel();

function preprocessImage(img) {
  return tf.tidy(() => {
    let tensor = tf.browser.fromPixels(img)
      .resizeBilinear([224, 224])
      .toFloat()
      .div(tf.scalar(255))
      .expandDims();
    return tensor;
  });
}

async function detect() {
  const file = document.getElementById("imageInput").files[0];
  if (!file) return alert("Please select an image.");

  const reader = new FileReader();
  reader.onload = async function (e) {
    const img = new Image();
    img.src = e.target.result;
    img.onload = async function () {
      document.getElementById("imagePreview").innerHTML = '';
      document.getElementById("imagePreview").appendChild(img);

      const input = preprocessImage(img);
      const prediction = await model.predict(input).data();
      const score = prediction[0]; // assuming 1 output neuron
      
      const label = score > 0.5 ? "🧪 Fake" : "✅ Real";
      const percentage = (score > 0.5 ? score : 1 - score) * 100;

      resultEl.textContent = `${label} — Confidence: ${percentage.toFixed(2)}%`;

      tf.dispose([input]);
    };
  };
  reader.readAsDataURL(file);
}
