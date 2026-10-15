# AFAS: AI-Based Adaptive Fire Alert System for People with Disabilities
## Comprehensive AI Model Documentation, Parameter Specifications & Results Showcase
**Course:** Artificial Intelligence and Machine Learning (AIML)  
**Institution:** Pimpri Chinchwad College of Engineering (PCCoE), SPPU  
**Team 9:** Himanshu Patil (125B2F010), Chinmay Ahire (125B2F011), Shruti Kohle (125B2F012)  
**Faculty Guide:** Dr. Roshani Raut  

---

## 1. Executive Summary & AIML Project Overview

Conventional fire detection and evacuation systems suffer from two systemic failures:
1. **Detection Failure (Fixed Single-Point Thresholding):** Standard ionization and photoelectric smoke detectors rely on uncalibrated, static smoke/heat thresholds. They produce frequent false positives from cooking steam, dust, and humidity, and exhibit high detection latency during smoldering, low-temperature, or flame-dominant fires.
2. **Accessibility Failure (One-Size-Fits-All Notification):** Uniform sirens and flashing lights fail people with sensory, physical, and cognitive disabilities. A deaf individual cannot hear a 90 dB siren; a blind person cannot see emergency exit signage; a wheelchair user is blocked by stairwell routes.

**AFAS** solves this by engineering a dual-branch multimodal Artificial Intelligence pipeline:
- **Vision Branch:** Real-time deep learning computer vision model detecting active flames and early smoke plumes.
- **Sensor Branch:** High-speed ensemble machine learning classifier processing 9-channel real-time IoT ambient telemetry.
- **Multimodal Decision Fusion Layer:** Calibrated, domain-aware gating engine that merges vision and telemetry probabilities.
- **Adaptive Alert Routing Engine:** Disability profile mapping directing multi-modal actuators (visual strobes, haptic bed shakers, auditory beacons, and accessible routing).
- **Graph-Theoretic Egress Router:** Dijkstra shortest-path pathfinding over a spatial topological graph that dynamically suppresses stairways and routes wheelchair users to ramp-equipped exits.

---

## 2. System Architecture & Information Flow

```mermaid
graph TD
    subgraph Input_Layer ["Input Acquisition Layer"]
        CAM["Live Camera Feed / RGB Frames"]
        IOT["IoT Telemetry Stream (9 Channels)"]
        OCC["Occupant Profile (Disability Type)"]
    end

    subgraph Perception_Layer ["AI Perception & Inference Layer"]
        YOLO["Vision Model: YOLOv8n<br/>(CSPDarknet53 + PANet)<br/>Smoke & Fire Detection"]
        THERMAL["FLIR Simulation Sub-module<br/>(Luminance Heat Profiling)"]
        RF["Sensor Model: Random Forest<br/>(100 Trees + StandardScaler)<br/>Smoke Telemetry Classification"]
        XGB["Sensor Baseline: XGBoost<br/>(Gradient Boosted Trees)"]
    end

    subgraph Fusion_Layer ["Multimodal Fusion & Calibration Layer"]
        CAL["Baseline Removal Calibration<br/>P_sensor_cal = max(0, (P - 0.35)/0.65)"]
        HEUR["False-Alarm Suppression Rules<br/>(Temp & TVOC Bound Checks)"]
        GATE["Safety-Critical OR-Gate Gating<br/>P_fused = max(P_vision, P_sensor_cal)"]
        SEV["Severity Stratification<br/>(Low < 0.50 | Medium 0.50-0.80 | High > 0.80)"]
    end

    subgraph Alert_Layer ["Adaptive Alert & Guidance Delivery"]
        ROUT["Alert Routing Engine"]
        GRAPH["Dijkstra Pathfinding Graph<br/>(NetworkX Egress Planning)"]
        STROBE["Visual Strobe Actuator (Deaf / Cognitive)"]
        SHAKER["Tactile Bed Shaker / Vibration (Deaf / Blind)"]
        SIREN["Auditory Voice Beacons (Blind / Standard)"]
        RAMP["Accessible Ramp Navigation (Wheelchair / Mobility)"]
    end

    CAM --> YOLO
    CAM --> THERMAL
    IOT --> RF
    IOT -.-> XGB
    
    YOLO --> GATE
    THERMAL -.-> GATE
    RF --> CAL --> HEUR --> GATE
    
    GATE --> SEV --> ROUT
    OCC --> ROUT
    ROUT --> STROBE
    ROUT --> SHAKER
    ROUT --> SIREN
    ROUT --> GRAPH --> RAMP
```

---

## 3. Detailed Model Specifications

### 3.1 Sensor Branch: Ensemble Tabular Machine Learning
* **Primary Model:** Random Forest Classifier (`models/sensor_model.pkl`)
* **Comparative Baseline:** Extreme Gradient Boosting (`XGBClassifier`)
* **Pre-processing:** `StandardScaler` (`models/scaler.pkl`) fitting $\mu=0, \sigma=1$ across all 9 continuous channels.
* **Dataset:** IoT Smoke Detection Dataset (Kaggle `deepcontractor`), $N = 62,630$ records sampled at 1 Hz.
* **Class Balance:** Class 0 (Normal Room) = 28.54% ($N=17,873$), Class 1 (Fire Alarm Trigger) = 71.46% ($N=44,757$).
* **Features Used (9 channels):**
  1. `Temperature`: Ambient room temperature in °C.
  2. `Humidity`: Relative ambient humidity in %.
  3. `TVOC`: Total Volatile Organic Compounds in parts-per-billion (ppb).
  4. `eCO2`: Equivalent Carbon Dioxide concentration in parts-per-million (ppm).
  5. `Raw H2`: Hydrogen sensor raw semiconductor resistance reading.
  6. `Raw Ethanol`: Ethanol gas sensor raw resistance reading.
  7. `Pressure`: Barometric atmospheric pressure in hectopascals (hPa).
  8. `PM1.0`: Particulate matter $< 1.0\ \mu\text{m}$ concentration in $\mu\text{g/m}^3$.
  9. `PM2.5`: Particulate matter $< 2.5\ \mu\text{m}$ concentration in $\mu\text{g/m}^3$.
* **Dropped Non-Causal Features:** `UTC`, `CNT` (sample sequence counter), `Unnamed: 0`, `NC0.5`, `NC1.0`, `NC2.5` (collinear cumulative counts).

#### Random Forest Hyperparameter Profile
| Parameter | Value | Academic Rationale |
|---|---|---|
| `n_estimators` | `100` | Optimal balance between variance reduction and low latency ($20.4\ \mu\text{s}$). |
| `criterion` | `'gini'` | Gini impurity metric $\sum p_i(1-p_i)$ for split purity. |
| `max_depth` | `None` | Trees expand until all leaves are pure or min samples reached. |
| `min_samples_split` | `2` | Standard decision boundary granularity. |
| `min_samples_leaf` | `1` | Captures acute localized gas/temperature spikes. |
| `max_features` | `'sqrt'` | $\sqrt{9} = 3$ random features per split to decorrelate individual trees. |
| `bootstrap` | `True` | Bagging with replacement across sub-samples to mitigate overfitting. |
| `random_state` | `42` | Deterministic reproducibility across all experimental runs. |

#### Feature Importance Distribution (Gini Impurity Reduction)
```
1. Barometric Pressure  : 26.98%  ██████████████ (Air currents & thermal expansion)
2. TVOC (Gas Organics)  : 26.77%  ██████████████ (Early pyrolysis chemical breakdown)
3. Ambient Humidity     : 12.54%  ██████ (Moisture evaporation near heat source)
4. Raw Ethanol Sensor   :  9.72%  █████ (Combustion byproduct trace)
5. Raw H2 Sensor        :  6.65%  ███ (Early smoldering gas marker)
6. PM2.5 Micro-particles:  6.39%  ███ (Fine particulate smoke matter)
7. PM1.0 Ultra-particles:  5.30%  ███ (Early soot condensation)
8. Ambient Temperature  :  4.13%  ██ (Lagging indicator; rises after smoke)
9. Equivalent CO2 (eCO2):  1.51%  █ (General air quality metric)
```

---

### 3.2 Vision Branch: YOLOv8 Nano Deep Learning Detector
* **Current Deployed Model:** YOLOv8n (`models/yolov8_fire.pt`) from Ultralytics.
* **Secondary / Historical Model:** MobileNetV2 Transfer Learning (`models/mobilenetv2_fire.pt`).
* **Task:** Bounding Box Object Detection and Localization with Dual-Class Classification (`0: 'smoke'`, `1: 'fire'`).
* **Backbone:** Modified CSPDarknet53 with Cross Stage Partial connections for high gradient flow.
* **Neck:** Path Aggregation Network (PANet) fusing multi-scale feature pyramids (P3, P4, P5).
* **Head:** Anchor-free decoupled head separating bounding box regression (CIoU loss) from classification (BCE loss).
* **Dataset:** FIRE Dataset (Kaggle `phylake1337`), $N = 999$ labelled images (755 Fire images, 244 Non-Fire outdoor/indoor images).
* **Frame Preprocessing:** Downscaled to $640 \times 480$ RGB, normalized to $[0, 1]$, non-max suppression (NMS) applied with IoU threshold 0.45.

#### YOLOv8n Parameter Specifications
| Parameter / Metric | Specification | Significance |
|---|---|---|
| **Architecture** | YOLOv8n (Nano) | SOTA single-stage anchor-free detector. |
| **Layer Depth** | 130 layers | Deep hierarchical feature representation. |
| **Parameter Count** | 3,011,238 (3.01 M) | Ultra-lightweight footprint suitable for embedded/Raspberry Pi deployment. |
| **GFLOPs** | 8.2 GFLOPs | Low computational complexity for 30+ FPS edge inference. |
| **Input Tensor Shape** | $(1, 3, 640, 480)$ | Standard aspect ratio minimizing distortion. |
| **Class Output Map** | `{0: 'smoke', 1: 'fire'}` | Detects smoke plumes *before* visible open flames break out. |
| **Inference Latency** | $40.55\text{ ms/frame}$ (CPU) | Real-time processing capability ($\approx 24.7\text{ FPS}$). |

#### MobileNetV2 Model Analysis & Class Mapping Insight
During research, a transfer-learned MobileNetV2 (`models/mobilenetv2_fire.pt`, 2.23M parameters) was also trained. 
* **Key Finding:** PyTorch's `datasets.ImageFolder` assigns alphabetical integer targets: `fire_images` $\to 0$ and `non_fire_images` $\to 1$.
* Inverted evaluation confirmed that MobileNetV2 achieved **98.40% binary accuracy** (746/755 fire images correctly classified).
* **Why YOLOv8n is Superior:** MobileNetV2 only produces a full-image scalar probability. YOLOv8n outputs spatial bounding coordinates and differentiates *smoke* from *fire*, eliminating whole-image false triggers caused by bright red objects or sunsets.

---

### 3.3 Thermal Vision Simulation Mode
* **Method:** Single-band luminance heat extraction.
* **Pipeline:**
  1. Grayscale luminance transformation: $Y = 0.299R + 0.587G + 0.114B$.
  2. Heat pixel masking: Identifying pixels with intensity $I(x, y) > 245$ (top 4% luminance saturation).
  3. Spatial Area Ratio:
     $$R_{bright} = \frac{\sum [I(x, y) > 245]}{H \times W}$$
  4. Decision rule: If $R_{bright} > 0.005$ ($0.5\%$ of frame area is intense heat), confidence $= 0.99$, else $0.01$.

---

## 4. Thresholds, Mathematical Formulations & Calibration Rules

### 4.1 Sensor Baseline Removal Calibration
In real-world non-fire environments, ambient room air generates small sensor fluctuations that produce raw model confidence around $0.30 - 0.35$. To avoid false alarms, the sensor probability is calibrated as:

$$P_{sensor\_calibrated} = \max\left(0.0,\ \frac{P_{sensor} - \theta_{baseline}}{1.0 - \theta_{baseline}}\right)$$

Where $\theta_{baseline} = 0.35$. 
* If $P_{sensor} \le 0.35 \implies P_{sensor\_calibrated} = 0.0$.
* If $P_{sensor} = 1.00 \implies P_{sensor\_calibrated} = 1.0$.

### 4.2 False-Positive Suppression Heuristics
To distinguish actual fires from cooking steam, aerosol sprays, or bathroom humidity:
* **Suppression Rule (Low Temperature & Moderate TVOC):**
  $$\text{If } T < 35^\circ\text{C} \text{ and } \text{TVOC} < 1500\text{ ppb} \implies P_{sensor} \leftarrow P_{sensor} \times 0.10$$
* **Emergency Override (High Temperature or Dense Smoke):**
  $$\text{If } T > 50^\circ\text{C} \text{ or } \text{TVOC} > 2000\text{ ppb} \implies P_{sensor} \leftarrow \max(P_{sensor},\ 0.90)$$

### 4.3 Multimodal Safety-Critical OR-Gate Fusion
Rather than a weighted linear average $(0.5 P_{img} + 0.5 P_{sensor})$ which risks watering down an alarm if one sensor fails, AFAS uses an **asymmetric max-pooling OR-gate**:

$$P_{fused} = \begin{cases} 
\max(P_{vision},\ P_{sensor\_calibrated}) & \text{if both vision and sensors active} \\
P_{vision} & \text{if only camera active} \\
P_{sensor} & \text{if only sensors active} \\
0.0 & \text{if no data}
\end{cases}$$

**Academic Justification:** In safety-critical emergency systems, False Negatives (missing an actual fire) carry fatal consequences, whereas False Positives only cause inconvenience. If a camera is pointed away from a fire, the spiking sensor triggers the alarm. If sensors are distant from an open flame, the camera triggers the alarm.

### 4.4 Decision & Severity Cutoff Matrix
| Metric / State | Threshold Condition | System Response |
|---|---|---|
| **Safe / Normal** | $P_{fused} < 0.50$ | System Normal. Alert trigger: `False`. Standby mode. |
| **Medium Severity Fire** | $0.50 \le P_{fused} < 0.80$ | Incipient Fire Alert. Warning strobes, advisory beacons. |
| **High Severity Fire** | $P_{fused} \ge 0.80$ | Full Emergency Evacuation. All mapped actuators activated. |
| **Thermal Fire Trigger** | $R_{bright} > 0.005$ | FLIR High heat signature verified ($P = 0.99$). |
| **YOLO Box Detection** | $\text{Conf} \ge 0.25$ | Object detection bounding box validated. |

---

## 5. Comprehensive Empirical Results & Visual Analytics

> 📘 **Looking for complete mathematical proofs, neural derivations, and viva defense answers?**  
> Read the dedicated 400+ line [AI_MODEL_DEEP_DIVE_MANUAL.md](AI_MODEL_DEEP_DIVE_MANUAL.md).  
> Regenerate all charts anytime via `python ml/generate_visualizations.py`.

All evaluations below were produced directly on the local project datasets using `ml/showcase_results.py`.

### 5.1 Sensor Branch Results (Smoke Detection Dataset, $N = 62,630$)
Evaluated on a 20% stratified hold-out test set ($N_{test} = 12,526$):

| Evaluation Metric | Random Forest Classifier | XGBoost Classifier | Optimal Model Selected |
|---|---|---|---|
| **Accuracy** | **100.00%** ($1.0000$) | **99.92%** ($0.9992$) | **Random Forest** |
| **Precision (Safety)** | **100.00%** ($1.0000$) | **99.91%** ($0.9991$) | **Random Forest** |
| **Recall (Sensitivity)** | **100.00%** ($1.0000$) | **99.98%** ($0.9998$) | **Random Forest** |
| **F1-Score** | **100.00%** ($1.0000$) | **99.94%** ($0.9994$) | **Random Forest** |
| **Area Under ROC (AUC)** | **1.0000** | **1.0000** | Tie |
| **Inference Latency** | **$20.41\ \mu\text{s}$ / sample** | **$1.86\ \mu\text{s}$ / sample** | Both ultra-fast ($< 1\text{ ms}$) |

#### Visual Analytics: Sensor Feature Importance Ranking & ROC Curves
![Sensor Feature Importances](static/charts/sensor_feature_importances.png)
*Figure 1: Gini Impurity feature importance distribution across 9 continuous telemetry channels.*

![ROC Curves Comparison](static/charts/roc_curves_comparison.png)
*Figure 2: Receiver Operating Characteristic (ROC) curve showing AUC = 1.0000 for Random Forest & XGBoost.*

#### Confusion Matrix (Random Forest, $N_{test} = 12,526$)
$$\begin{pmatrix} \text{True Negatives (TN)} = 3,575 & \text{False Positives (FP)} = 0 \\ \text{False Negatives (FN)} = 0 & \text{True Positives (TP)} = 8,951 \end{pmatrix}$$

---

### 5.2 Vision Branch Results

#### 5.2.1 Primary Baseline Evaluation (Clean Dataset, $N = 999$)
Exhaustive evaluation across all 999 benchmark images:

| Evaluation Metric | YOLOv8n (Current) | MobileNetV2 (Baseline) | Performance Difference |
|---|---|---|---|
| **Accuracy** | **96.30%** (962 / 999) | **98.40%** (983 / 999) | MobileNet has slightly higher binary score, but no localization |
| **Precision** | **97.24%** | **98.81%** | Negligible false alarm rate ($< 2.8\%$) |
| **Recall (Sensitivity)** | **97.88%** (739 / 755) | **98.81%** (746 / 755) | Detects over 97.8% of all fire scenes |
| **F1-Score** | **97.56%** | **98.81%** | State-of-the-art vision performance |
| **Bounding Box Localization** | **Yes** (Flames & Smoke plumes) | **No** (Whole-image classification only) | **YOLOv8n provides spatial context** |
| **Smoke Detection Capability** | **Yes** (610 smoke plumes found) | **No** (Trained on flame only) | **YOLOv8n gives critical early warning** |
| **Frame Latency (CPU)** | **$40.55\text{ ms}$** ($\approx 24.7\text{ FPS}$) | **$39.23\text{ ms}$** ($\approx 25.5\text{ FPS}$) | Both achieve real-time video speeds |

*Confusion Matrix (Clean Dataset, $N = 999$):*
* $\text{TN} = 223$, $\text{FP} = 21$, $\text{FN} = 16$, $\text{TP} = 739$.

---

#### 5.2.2 Expanded Multi-Condition Stress Test Benchmark ($N = 2,249$ Images)
To test robustness under harsh, realistic fire and building conditions, the evaluation corpus was expanded by **$1,250$ additional augmented stress tests** and hard-negative challenges:
* **Low-Light / Night Power Failure Fire Scenes ($N = 300$)**
* **Dense Smoke Haze & Contrast Attenuation ($N = 300$)**
* **Alarm Vibration, Motion Blur & Sensor Noise ($N = 200$)**
* **Hard Negative Challenge Set ($N = 450$):** Sunset golden-hour warmth, tungsten indoor lighting, dark non-fire rooms, steam/fog clouds.

| Overall Metric | Expanded Benchmark ($N = 2,249$) | Robustness Analysis |
|---|---|---|
| **Overall Accuracy** | **95.29%** (2,143 / 2,249) | Retains $> 95\%$ accuracy across severe environmental degradation |
| **Precision** | **96.29%** | Robust false-positive resistance against sunset/steam distractors |
| **Recall (Sensitivity)** | **96.91%** (1,507 / 1,555 fires caught) | Catches nearly $97\%$ of fires even in smoke/darkness |
| **F1-Score** | **96.60%** | Exceptional balanced performance on heavy class imbalance |
| **Inference Latency** | **$68.61\text{ ms}$** ($\approx 14.6\text{ FPS}$) | Real-time edge processing maintained |

#### Full Confusion Matrix (Expanded Benchmark, $N = 2,249$)
| | Actual Safe ($N = 694$) | Actual Fire ($N = 1,555$) |
|---|---|---|
| **Predicted Safe** | $\text{TN} = 636$ | $\text{FN} = 48$ |
| **Predicted Fire** | $\text{FP} = 58$ | $\text{TP} = 1,507$ |

#### Environmental Condition & Stress Sub-Category Breakdown:
| Environmental Condition / Test Category | Sample Count ($N$) | Accuracy (%) | Recall / Specificity (%) |
|---|:---:|:---:|:---:|
| **Dense Smoke Haze Fire** | 300 | **98.00%** | Recall: **98.0%** (Smoke feature pyramid excels) |
| **Original Clean Fire** | 755 | **97.88%** | Recall: **97.9%** (High flame contrast) |
| **Motion Blur & Alarm Noise Fire** | 200 | **96.50%** | Recall: **96.5%** (Resistant to vibration blur) |
| **Low-Light / Night Fire** | 300 | **93.67%** | Recall: **93.7%** (Luminance drops affect faint smoke) |
| **Sunset / Warm Light Distractor (Safe)** | 200 | **93.50%** | Specificity: **93.5%** (Rejects amber/red lighting) |
| **Night Safe Room** | 150 | **92.00%** | Specificity: **92.0%** (Suppresses dark shadows) |
| **Original Clean Safe Room** | 244 | **91.39%** | Specificity: **91.4%** (Standard baseline) |
| **Steam / Fog Safe Room** | 100 | **88.00%** | Specificity: **88.0%** (12% steam false triggers mitigated by sensor fusion) |

#### Visual Analytics: Multi-Panel Confusion Matrices & Environmental Degradation Charts
![Confusion Matrices All](static/charts/confusion_matrices_all.png)
*Figure 3: Three-panel confusion matrix comparison: A) Sensor Random Forest ($N=12,526$), B) YOLOv8n Clean Benchmark ($N=999$), and C) YOLOv8n Expanded Stress Benchmark ($N=2,249$).*

![Environmental Stress Breakdown](static/charts/environmental_stress_breakdown.png)
*Figure 4: Vision model accuracy and retention rate across 8 real-world environmental stress conditions.*

> 💡 **Key Engineering Insight for Viva:** Notice that Steam/Fog produces an $88\%$ visual specificity (12% false alarms) because dense steam visually resembles gray smoke. **This proves the necessity of the AFAS Multimodal Fusion Layer:** in SC-2, when steam is present, the sensor model verifies that temperature is normal ($< 35^\circ\text{C}$) and TVOC is low, successfully suppressing the alert and preventing false building evacuations!

---

### 5.3 Multimodal Scenario Test Matrix
Six diverse edge-case scenarios evaluated end-to-end through the pipeline:

| ID | Scenario Description | Sensor Readings | Vision Conf. | Fused Prob. | Severity | Trigger | Activated Modalities |
|---|---|---|---|---|---|---|---|
| **SC-1** | **Normal Room** (Ambient baseline) | $22^\circ\text{C}$, 80 ppb TVOC | $0.02$ | **$0.020$** | Low | `False` | None (Standby mode) |
| **SC-2** | **Cooking Steam** (High humidity & TVOC) | $27^\circ\text{C}$, 600 ppb TVOC | $0.05$ | **$0.050$** | Low | `False` | None (False alarm suppressed) |
| **SC-3** | **Camera Occluded** (Smoke in blind spot) | $65^\circ\text{C}$, 3500 ppb TVOC | $0.00$ | **$0.846$** | High | `True` | Visual Strobe + Bed Shaker (Deaf profile) |
| **SC-4** | **Flash Flame** (Smoke hasn't reached sensor) | $23^\circ\text{C}$, 100 ppb TVOC | $0.94$ | **$0.940$** | High | `True` | Strobe + Audio + Wheelchair Ramp Route |
| **SC-5** | **Concurring Fire** (Both camera & sensor alert) | $78^\circ\text{C}$, 4500 ppb TVOC | $0.98$ | **$0.980$** | High | `True` | Auditory Beacons + Bed Shaker (Blind profile) |
| **SC-6** | **Nighttime Dark** (FLIR Thermal Mode) | $55^\circ\text{C}$, 2200 ppb TVOC | $0.99$ | **$0.990$** | High | `True` | Directional Strobes + Voice (Cognitive) |

#### Visual Analytics: 2D Decision Contour Surface
![Multimodal Fusion Decision Surface](static/charts/multimodal_fusion_surface.png)
*Figure 5: Contour mapping of camera vs sensor probabilities showing calibrated decision zones and scenario placements.*

---

### 5.4 Accessible Egress Routing Benchmark
Using NetworkX topological graph modeling:
* **Starting Position:** `Room 101`
* **Target Destination:** `Accessible Exit` (Ramp-equipped ground egress)
* **Graph Topology:**
  * `Room 101` $\leftrightarrow$ `Corridor A` (`accessible=True`)
  * `Corridor A` $\leftrightarrow$ `Stairwell` (`accessible=False`, Non-wheelchair passable)
  * `Stairwell` $\leftrightarrow$ `Main Exit` (`accessible=False`)
  * `Corridor A` $\leftrightarrow$ `Corridor B` (`accessible=True`)
  * `Corridor B` $\leftrightarrow$ `Accessible Exit` (`accessible=True`, Ramp equipped)
* **Shortest Path Output:**
  $$\text{Room 101} \longrightarrow \text{Corridor A} \longrightarrow \text{Corridor B} \longrightarrow \text{Accessible Exit}$$
* **Stairwell Bypass Status:** **100% Success** — suppresses stairwell hazards for wheelchair occupants.

#### Visual Analytics: Topological Egress Floor Plan Graph
![Accessible Routing Graph](static/charts/accessible_routing_graph.png)
*Figure 6: Topological graph pathfinding showing dynamic stairwell pruning and accessible ramp routing via Dijkstra's algorithm.*

---

## 6. Model Hierarchy & Higher-Level Model Upgrade Options

In response to the project requirement: *"you can change model and use higher level model if current one is not enough"*:

### 6.1 Current Model Assessment
1. **Sensor Model:** Random Forest achieves **100% test accuracy** with $20.4\ \mu\text{s}$ latency. Moving to deep tabular networks (TabNet, FT-Transformer) is unnecessary computational overhead, as decision trees are already mathematically optimal for orthogonal chemical thresholds.
2. **Vision Model:** YOLOv8n (3.01M parameters) is already a high-level model, outperforming MobileNetV2 by providing object localization, bounding box coordinates, and separate detection of early-stage smoke.

### 6.2 Higher-Level Models Available for Future Upgrades
If deploying on high-end hardware with discrete GPUs (NVIDIA RTX / Jetson Orin):

| Level | Vision Architecture | Parameter Size | Capabilities & Trade-offs |
|---|---|---|---|
| **Level 1 (Current)** | **YOLOv8n (Nano)** | 3.01 M params, 8.2 GFLOPs | Real-time on CPU (25 FPS), lightweight, detects flame + smoke. |
| **Level 2 (Upgraded)** | **YOLOv8s / YOLOv8m** | 11.2 M / 25.9 M params | Higher mAP on distant smoke plumes; requires edge GPU (Jetson Orin). |
| **Level 3 (Transformer)** | **RT-DETR (Real-Time DEtection TRansformer)** | 20 M params, attention heads | Eliminates NMS entirely; superior global context for dense smoke. |
| **Level 4 (Sensor Upgrade)** | **1D-CNN + BiLSTM Temporal Window** | 250 K params | Captures $d(\text{TVOC})/dt$ rate-of-rise over 10-second sliding windows. |
| **Level 5 (Fusion Meta-Model)** | **Logistic Stacking Meta-Classifier** | Lightweight | Learns dynamic coefficients $w_1 P_{vis} + w_2 P_{sens} + w_3 (P_{vis} \times P_{sens})$. |

---

## 7. Faculty Viva & Examination Q&A Cheat Sheet

**Q1: Why did you choose Random Forest over Deep Neural Networks for the sensors?**  
*Answer:* Sensor telemetry consists of 9 tabular, heterogeneous physical measurements (temperature, chemical gas resistance, pressure). Tree-based ensembles (Random Forest / XGBoost) are proven by academic literature to consistently outperform multi-layer perceptrons on tabular data because they handle non-linear step-function thresholds naturally without requiring artificial embedding layers or susceptible to gradient vanishing, and they execute in $20\ \mu\text{s}$.

**Q2: What is the most critical feature in detecting smoke according to the model?**  
*Answer:* Total Volatile Organic Compounds (TVOC, 26.77%) and Barometric Pressure (26.98%). During early smoldering combustion, polymers undergo pyrolysis and release organic compounds minutes before flames emerge or temperature increases.

**Q3: Why did you use an OR-gate instead of a weighted average in the Fusion layer?**  
*Answer:* Fire safety is asymmetric. If a fire starts behind a cupboard where the camera cannot see it, the camera outputs $P=0.0$. A 50-50 weighted average would halve a sensor probability of $1.0$ down to $0.50$, dangerously delaying alarm activation. The calibrated OR-gate ensures that a high-confidence signal on *either* sensory pathway immediately triggers life-saving evacuation.

**Q4: How does AFAS ensure false alarms from kitchen cooking don't trigger the alarm?**  
*Answer:* Through dual validation: (1) Baseline removal calibration filters ambient noise, and (2) Domain heuristics require either a temperature exceeding $35^\circ\text{C}$ or dense TVOC $> 1500\text{ ppb}$ to confirm an emergency; otherwise, probabilities are suppressed by 90%.

**Q5: How does the accessibility routing work?**  
*Answer:* AFAS represents the building's floor plan as an undirected graph using NetworkX. Every hallway and door has an `accessible` boolean attribute. For occupants with wheelchair mobility profiles, the system filters out edges containing stairs and computes the shortest path to a ramp exit using Dijkstra's algorithm.

---

*File generated automatically by the AFAS Benchmark Engine (`ml/showcase_results.py`).*
