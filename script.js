document.addEventListener("DOMContentLoaded", () => {
    const status = document.getElementById("status");
    const result = document.getElementById("result");
    const fileInput = document.getElementById("file");
    const predictBtn = document.getElementById("predictBtn");

    // Dummy model load simulation
    function loadModel() {
        status.innerText = "Model loaded successfully!";
        console.log("Dummy model loaded");
    }

    // Fake prediction logic
    function predict() {
        const file = fileInput.files[0];
        if (!file) {
            alert("Please choose an image first!");
            return;
        }

        // Simulate prediction delay
        result.innerText = "Predicting...";
        setTimeout(() => {
            const random = Math.random();
            if (random > 0.5) {
                result.innerText = "Result: FAKE IMAGE 😵‍💫";
                result.style.color = "red";
            } else {
                result.innerText = "Result: REAL IMAGE 🧑‍🦱";
                result.style.color = "green";
            }
        }, 1500);
    }

    // Attach event
    predictBtn.addEventListener("click", predict);

    // Simulate model loading on page load
    loadModel();
});
