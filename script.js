let model;

async function loadModel() {
  try {
    document.getElementById("result").innerText = "Loading model...";
    model = await tf.loadLayersModel("model.json");
    document.getElementById("result").innerText = "Model loaded! Upload an image.";
  } catch (err) {
    document.getElementById("result").innerText = "Failed to load model.";
    console.error("Model load error:", err);
  }
}

loadModel();

function preprocessImage(image) {
  return tf.tidy(() => {
    return tf.browser.fromPixels(image)
      .resizeNearestNeighbor([224, 224]) // Assumes your model uses 224x224 input
      .toFloat()
      .div(255.0)
      .expandDims();
  });
}

async function detect() {
  const fileInput = document.getElementById("imageInput");
  const file = fileInput.files[0];
  if (!file) {
    alert("Please upload an image first!");
    return;
  }

  const img = new Image();
  img.src = URL.createObjectURL(file);
  img.onload = async () => {
    document.getElementById("preview").src = img.src;

    const input = preprocessImage(img);
    const prediction = await model.predict(input).data();
    const probability = prediction[0];

    let label = "";
    if (probability > 0.5) {
      label = `🔴 Fake Face Detected — Confidence: ${(probability * 100).toFixed(2)}%`;
    } else {
      label = `🟢 Real Face Detected — Confidence: ${((1 - probability) * 100).toFixed(2)}%`;
    }

    document.getElementById("result").innerText = label;
  };
}
