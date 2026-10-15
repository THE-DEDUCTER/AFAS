<div align="center">

# 🔥 AFAS
### AI-Based Adaptive Fire Alert System for People with Disabilities

*An ML-powered fire detection system that adapts its alert — visual, vibration, or audio — to the person who needs it.*

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=flat&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-EB0028?style=flat)
![Status](https://img.shields.io/badge/status-in%20development-yellow?style=flat)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat)

</div>

---

## 🧭 Overview

Standard fire alarms assume everyone can hear a siren, see a strobe, and take the stairs. For wheelchair users, the deaf and hard-of-hearing, the blind and visually impaired, and cognitively impaired individuals, that assumption can be fatal.

**AFAS** detects fire in real time using both a **camera feed** and **environmental sensor data**, fuses the two signals into a single decision, and triggers a **multi-modal, occupant-specific alert** — instead of one generic siren for everyone.

> 📊 This project is built entirely in **Python** — ML models, backend, and UI — with no separate frontend framework or database.

---

## ✨ Features

- 🖼️ **Image-based fire detection** — CNN (MobileNetV2, transfer-learned) classifies camera frames as fire / no-fire
- 📡 **Sensor-based fire detection** — Random Forest / XGBoost classifies live temperature, smoke, and gas readings
- 🔗 **Fusion layer** — combines both predictions into one confident decision
- 🚨 **Adaptive alert engine** — maps a registered occupant's disability profile to the right alert combination (visual strobe / vibration / audio)
- 🖥️ **Live demo dashboard** — upload an image or enter sensor values and watch the full pipeline run, with the resulting alert simulated on-screen
- 📈 **Model evaluation** — accuracy, precision, recall, F1-score, and confusion matrix for both branches

---

## 🏗️ Architecture

```
        Camera Feed                Sensor Stream
             │                          │
             ▼                          ▼
      ┌─────────────┐           ┌──────────────┐
      │ Image Branch│           │ Sensor Branch│
      │  (CNN)      │           │ (RF/XGBoost) │
      └──────┬──────┘           └──────┬───────┘
             │                          │
             └───────────┬──────────────┘
                          ▼
                  ┌───────────────┐
                  │ Fusion Layer  │
                  └───────┬───────┘
                          ▼
                ┌───────────────────┐
                │ Alert Routing     │
                │ Engine (occupant  │
                │ profile → alert)  │
                └─────────┬─────────┘
                          ▼
        ┌─────────┬─────────┬─────────┬──────────┐
        ▼         ▼         ▼         ▼
     Strobe   Vibration    Audio   Accessible-Exit
                                      Guidance
```

> 🧩 **Note:** The web dashboard is a **pipeline visualizer for the demo** — not the real-world alert delivery mechanism. In a production deployment, alerts would drive physical hardware (strobe lights, bed shakers, smart-home sirens) rather than a webpage.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Image model | PyTorch / TensorFlow + MobileNetV2 |
| Sensor model | scikit-learn (Random Forest) + XGBoost |
| Backend & Web server | Flask |
| UI Templates | Jinja2 |
| Styling | Tailwind CSS (CDN) |
| Interactivity | Vanilla JS (`fetch`) |
| Data handling | pandas, numpy, Pillow/OpenCV |
| Dev tools | VS Code / Antigravity, Git |

---

## 📂 Project Structure

```
afas/
├── app.py                  # Flask app — routes + API endpoints
├── ml/
│   ├── image_model.py      # CNN inference
│   ├── sensor_model.py     # RF/XGBoost inference
│   ├── fusion.py           # combines both predictions
│   └── alert_engine.py     # occupant profile → alert mapping
├── train/
│   ├── train_image_model.py
│   └── train_sensor_model.py
├── data/
│   ├── raw/                # original datasets
│   └── processed/          # cleaned/augmented data
├── models/                 # saved trained weights
├── templates/               # Jinja2 HTML templates
│   ├── base.html
│   ├── dashboard.html
│   └── components/
│       └── alert_card.html
├── static/
│   ├── css/
│   └── js/
│       └── dashboard.js
├── requirements.txt
└── README.md
```

---

## 📊 Datasets

| Dataset | Source | Size | Used For |
|---|---|---|---|
| [FIRE Dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset) | Kaggle (`phylake1337`) | 999 images (755 fire / 244 non-fire) | Image branch |
| [Smoke Detection Dataset](https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset) | Kaggle (`deepcontractor`) | ~60,000 sensor readings @ 1Hz | Sensor branch |

> Neither dataset is disability-labelled — the occupant-profile → alert-type mapping is a rule-based layer defined by the team, not learned from data.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- `pip`

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/afas.git
cd afas

# Create a virtual environment
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download the datasets
1. Download the [FIRE Dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset) and place it under `data/raw/fire_dataset/`
2. Download the [Smoke Detection Dataset](https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset) and place it under `data/raw/smoke_detection/`

### Train the models

```bash
python train/train_image_model.py
python train/train_sensor_model.py
```

### Run the app

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## 📈 Model Evaluation

| Model | Task & Architecture | Accuracy | Precision | Recall | F1-Score | Latency |
|---|---|---|---|---|---|---|
| **Vision — YOLOv8n (Expanded)** | Multi-Condition Stress Test ($N=2,249$) | **95.29%** | **96.29%** | **96.91%** | **96.60%** | $68.6\text{ ms}$ (14.6 FPS) |
| **Vision — YOLOv8n (Clean)** | Clean Benchmark ($N=999$) | **96.30%** | **97.24%** | **97.88%** | **97.56%** | $40.5\text{ ms}$ (24.7 FPS) |
| **Vision — MobileNetV2** *(Baseline)* | Binary Classification (CNN) | **98.40%** | **98.81%** | **98.81%** | **98.81%** | $39.2\text{ ms}$ (25.5 FPS) |
| **Sensor — Random Forest** *(Deployed)* | 9-Channel IoT Smoke Classification | **100.00%** | **100.00%** | **100.00%** | **100.00%** | $20.4\ \mu\text{s}$ |
| **Sensor — XGBoost** *(Baseline)* | Gradient Boosted Trees | **99.92%** | **99.91%** | **99.98%** | **99.94%** | $1.86\ \mu\text{s}$ |
| **Fused Pipeline (Vision + Sensors)** | Calibrated Asymmetric OR-Gating | **99.9%** | **High** | **100.0%** | **Optimal** | $< 45\text{ ms}$ End-to-End |

> 📑 **Full Technical Documentation & Parameter Breakdown:**  
> See the dedicated [MODEL_SHOWCASE_AND_RESULTS.md](MODEL_SHOWCASE_AND_RESULTS.md) for hyperparameter specifications, confusion matrices, environmental stress breakdown, edge-case scenarios, and academic viva Q&A. Run `python ml/showcase_results.py` or `python ml/evaluate_expanded_vision.py` to regenerate all benchmarks.

---

## 🗺️ Roadmap

- [x] Problem definition & literature survey
- [x] Dataset selection
- [ ] Train image & sensor models
- [ ] Build fusion layer
- [ ] Build alert routing engine
- [ ] Build Flask dashboard demo
- [ ] Model evaluation & report
- [ ] (Stretch) Accessible-exit route guidance

---

## 👥 Team

| Name | PRN |
|---|---|
| Himanshu Liladhar Patil | 125B2F010 |
| Chinmay Ravindra Ahire | 125B2F011 |
| Shruti Ravindra Kohle | 125B2F012 |

**Guided by:** Dr. Roshani Raut
**Course:** Artificial Intelligence and Machine Learning — TY IT, PCCoE (SPPU), 2026–27 Sem V

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
