const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const predictBtn = document.getElementById("predictBtn");
const resultDiv = document.getElementById("result");

// Preview image
imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = e => {
      imagePreview.src = e.target.result;
      imagePreview.style.display = "block";
    };
    reader.readAsDataURL(file);
    resultDiv.innerText = ""; // clear previous result
  }
});

// Dummy Predict Function
predictBtn.addEventListener("click", () => {
  if (!imageInput.files.length) {
    alert("Please select an image first!");
    return;
  }

  // Simulate prediction
  const confidence = Math.floor(Math.random() * 41) + 60; // 60% to 100%
  const isReal = Math.random() < 0.5; // Randomly choose real or fake

  const resultText = isReal
    ? `✅ ${confidence}% Real`
    : `🚨 ${100 - confidence}% Fake`;

  resultDiv.innerText = resultText;
  resultDiv.style.color = isReal ? "#00ff88" : "#ff4d4d";
});
