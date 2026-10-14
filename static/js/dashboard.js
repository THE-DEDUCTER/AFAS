document.addEventListener("DOMContentLoaded", () => {
    const tempSlider = document.getElementById("temp");
    const tvocSlider = document.getElementById("tvoc");
    const tempVal = document.getElementById("temp-val");
    const tvocVal = document.getElementById("tvoc-val");
    const toggleBtn = document.getElementById("toggle-monitor");
    const webcam = document.getElementById("webcam");
    const canvas = document.getElementById("canvas");
    const webcamPlaceholder = document.getElementById("webcam-placeholder");
    
    let isMonitoring = false;
    let monitorInterval = null;
    let stream = null;
    // Alert UI elements
    const alertCard = document.getElementById("alert-card");
    const statusIcon = document.getElementById("status-icon");
    const alertMessage = document.getElementById("alert-message");
    const alertGuidance = document.getElementById("alert-guidance");
    const strobeOverlay = document.getElementById("strobe-overlay");
    const modalityBadges = document.getElementById("modality-badges");
    
    const logs = document.getElementById("logs");
    const themeToggle = document.getElementById("theme-toggle");

    // Set initial values from cached slider position (fixes browser refresh sync issue)
    tempVal.textContent = tempSlider.value;
    tvocVal.textContent = tvocSlider.value;

    // Dark mode toggle
    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            document.documentElement.classList.toggle("dark");
            if (document.documentElement.classList.contains("dark")) {
                themeToggle.textContent = "☀️ Light Mode";
            } else {
                themeToggle.textContent = "🌙 Dark Mode";
            }
        });
    }

    // Thermal Mode Toggle
    const thermalToggle = document.getElementById("thermal-toggle");
    let isThermalMode = false;

    if (thermalToggle) {
        thermalToggle.addEventListener("change", (e) => {
            isThermalMode = e.target.checked;
            if (isThermalMode) {
                // Fake thermal vision (Ironbow-style: dark is red/purple, bright is yellow/white)
                webcam.style.filter = "grayscale(1) sepia(1) hue-rotate(330deg) saturate(5) contrast(2)";
            } else {
                webcam.style.filter = "none";
            }
        });
    }

    // Update slider values on drag
    tempSlider.addEventListener("input", (e) => tempVal.textContent = e.target.value);
    tvocSlider.addEventListener("input", (e) => tvocVal.textContent = e.target.value);

    // Webcam setup
    async function startWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            webcam.srcObject = stream;
            webcam.classList.remove("hidden");
            webcamPlaceholder.classList.add("hidden");
        } catch (err) {
            console.error("Error accessing webcam: ", err);
            logMessage("Error: Please grant webcam permissions.");
        }
    }

    function stopWebcam() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            webcam.srcObject = null;
        }
        webcam.classList.add("hidden");
        webcamPlaceholder.classList.remove("hidden");
    }

    // Capture frame and predict
    async function sendPredictionRequest() {
        if (!isMonitoring) return;

        const occupant = document.getElementById("occupant").value;
        const sensorData = {
            "Temperature": tempSlider.value,
            "TVOC": tvocSlider.value
        };

        const formData = new FormData();
        formData.append("occupant_profile", occupant);
        formData.append("sensor_data", JSON.stringify(sensorData));
        formData.append("thermal_mode", isThermalMode);

        // Capture frame from video. Downscale to 640x480 to prevent MemoryError while keeping YOLOv8 accurate
        if (webcam.videoWidth > 0) {
            canvas.width = 640;
            canvas.height = 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);
            
            // Convert to Blob and append
            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
            if (blob) {
                formData.append("image", blob, "frame.jpg");
            }
        }

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });
            const data = await response.json();
            
            if (data.status === "success") {
                updateAlertUI(data.alert);
            }
        } catch (error) {
            console.error("Fetch Error:", error);
        }
    }

    // Toggle button logic
    toggleBtn.addEventListener("click", async () => {
        if (isMonitoring) {
            isMonitoring = false;
            clearInterval(monitorInterval);
            stopWebcam();
            toggleBtn.textContent = "Start Monitoring";
            toggleBtn.classList.remove("bg-gray-600", "hover:bg-gray-700");
            toggleBtn.classList.add("bg-red-600", "hover:bg-red-700");
            logMessage("Monitoring stopped.");
        } else {
            isMonitoring = true;
            await startWebcam();
            monitorInterval = setInterval(sendPredictionRequest, 1000); // 1 FPS
            toggleBtn.textContent = "Stop Monitoring";
            toggleBtn.classList.remove("bg-red-600", "hover:bg-red-700");
            toggleBtn.classList.add("bg-gray-600", "hover:bg-gray-700");
            logMessage("Monitoring started...");
        }
    });

    function updateAlertUI(alert) {
        // Reset state
        alertCard.classList.remove("danger-state", "alert-vibrate-active");
        strobeOverlay.classList.remove("alert-strobe-active");
        
        document.getElementById("badge-visual").classList.add("hidden");
        document.getElementById("badge-vibration").classList.add("hidden");
        document.getElementById("badge-audio").classList.add("hidden");

        if (alert.trigger) {
            // Danger state
            alertCard.classList.add("danger-state");
            statusIcon.textContent = "🔥";
            statusIcon.classList.add("scale-150", "text-red-600");
            
            alertMessage.textContent = alert.message;
            alertMessage.classList.add("text-red-600");
            
            alertGuidance.textContent = alert.guidance;
            alertGuidance.classList.remove("opacity-0");
            
            modalityBadges.classList.remove("opacity-0");

            // Apply specific modalities
            if (alert.visual) {
                strobeOverlay.classList.add("alert-strobe-active");
                document.getElementById("badge-visual").classList.remove("hidden");
            }
            if (alert.vibration) {
                alertCard.classList.add("alert-vibrate-active");
                document.getElementById("badge-vibration").classList.remove("hidden");
            }
            if (alert.audio) {
                document.getElementById("badge-audio").classList.remove("hidden");
            }
            
        } else {
            // Normal state
            statusIcon.textContent = "🛡️";
            statusIcon.classList.remove("scale-150", "text-red-600");
            
            alertMessage.textContent = alert.message;
            alertMessage.classList.remove("text-red-600");
            
            alertGuidance.classList.add("opacity-0");
            modalityBadges.classList.add("opacity-0");
        }
    }

    function logMessage(msg) {
        logs.classList.remove("hidden");
        const p = document.createElement("p");
        p.textContent = `> ${msg}`;
        logs.appendChild(p);
        logs.scrollTop = logs.scrollHeight;
    }
});
