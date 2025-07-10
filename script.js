let model;

async function loadModel() {
    document.getElementById("status").innerText = "Loading model...";
    try {
        model = await tf.loadLayersModel("model.json");  // update if in a subfolder
        document.getElementById("status").innerText = "Model loaded!";
    } catch (error) {
        document.getElementById("status").innerText = "Failed to load model.";
        console.error("Model load error:", error);
    }
}

async function predict() {
    const fileInput = document.getElementById("imageInput");
    if (!fileInput.files.length) {
        alert("Please choose an image.");
        return;
    }

    const img = await loadImage(fileInput.files[0]);
    const tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(255.0)
        .expandDims();

    const prediction = model.predict(tensor);
    const result = await prediction.data();

    const confidence = (result[0] * 100).toFixed(2);
    const label = result[0] > 0.5 ? "Fake" : "Real";

    document.getElementById("result").innerText = `Prediction: ${label} (${confidence}%)`;
}

function loadImage(file) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.src = URL.createObjectURL(file);
    });
}

window.addEventListener("load", loadModel);
