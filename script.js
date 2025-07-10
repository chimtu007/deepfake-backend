const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const detectBtn = document.getElementById("predictBtn");
const resultDiv = document.getElementById("result");

let hasPredicted = false; // lock prediction after one detect per image

// Reset and preview new image
imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = e => {
      imagePreview.src = e.target.result;
      imagePreview.style.display = "block";
      resultDiv.innerText = "";
      hasPredicted = false;
    };
    reader.readAsDataURL(file);
  }
});

// Dummy Detect Function (runs only once per image)
detectBtn.innerText = "🔍 Detect";

detectBtn.addEventListener("click", () => {
  if (!imageInput.files.length) {
    alert("Please select an image first!");
    return;
  }

  if (hasPredicted) return;

  // Fake fixed response
  const fixedConfidence = 83;
  const isReal = false; // Always fake in this dummy

  const resultText = isReal
    ? `✅ ${fixedConfidence}% Real`
    : `🚨 ${100 - fixedConfidence}% Fake`;

  resultDiv.innerText = resultText;
  resultDiv.style.color = isReal ? "#00ff88" : "#ff4d4d";

  hasPredicted = true;
});
