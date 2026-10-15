# AFAS: Comprehensive AI/ML Model Deep-Dive & Technical Reference Manual
## Complete Architectural, Mathematical, Parametric & Experimental Documentation
**Academic Course:** Artificial Intelligence and Machine Learning (AIML)  
**Institution:** Pimpri Chinchwad College of Engineering (PCCoE), SPPU  
**Project:** AI-Based Adaptive Fire Alert System for People with Disabilities (AFAS)  
**Authors (Team 9):** Himanshu Patil, Chinmay Ahire, Shruti Kohle  
**Project Mentor:** Dr. Roshani Raut  

---

## Table of Contents
1. [Mathematical Foundations & Problem Formulation](#1-mathematical-foundations--problem-formulation)
2. [Sensor Branch: Tabular Ensemble Machine Learning](#2-sensor-branch-tabular-ensemble-machine-learning)
   - 2.1 Physics & Chemical Sensing Principles of the 9 Channels
   - 2.2 Random Forest: Algorithmic Mechanics & Derivations
   - 2.3 XGBoost: Second-Order Gradient Boosting Formulation
   - 2.4 Hyperparameter Profiles & Search Spaces
   - 2.5 Feature Importance Mathematics (MDI vs MDA)
3. [Vision Branch: Deep Convolutional & Decoupled Object Detection](#3-vision-branch-deep-convolutional--decoupled-object-detection)
   - 3.1 YOLOv8n Architecture: Layer-by-Layer Breakdown
   - 3.2 Loss Function Formulations (BCE, CIoU, DFL)
   - 3.3 MobileNetV2 Architecture & Comparative Baseline
   - 3.4 FLIR Thermal Vision Simulation Physics
4. [Multimodal Fusion Engine & Calibration Theory](#4-multimodal-fusion-engine--calibration-theory)
   - 4.1 Ambient Baseline Removal Calibration
   - 4.2 Asymmetric Safety-Critical OR-Gate vs. Weighted Averaging
   - 4.3 Domain-Grounded False Positive Suppression Heuristics
5. [Graph-Theoretic Accessible Egress Planning](#5-graph-theoretic-accessible-egress-planning)
   - 5.1 Graph Representation & Adjacency Modeling
   - 5.2 Dijkstra Shortest-Path Algorithm & Edge Pruning
6. [Empirical Results Showcase & Visual Analytics](#6-empirical-results-showcase--visual-analytics)
   - 6.1 Gini Impurity Feature Importance Distribution
   - 6.2 Confusion Matrices (Sensors, Clean Vision, Stress-Test Vision)
   - 6.3 Environmental Robustness Breakdown Across 8 Conditions
   - 6.4 2D Multimodal Decision Surface & Scenario Validation
   - 6.5 ROC-AUC Characteristics
   - 6.6 Topological Graph Egress Visualization
7. [Hardware Requirements & Quantization Roadmap](#7-hardware-requirements--quantization-roadmap)
8. [Comprehensive Viva Voce & Technical Defense Guide (20 Questions)](#8-comprehensive-viva-voce--technical-defense-guide)

---

## 1. Mathematical Foundations & Problem Formulation

### 1.1 Problem Statement in AIML Formalism
Let $\mathcal{X}_{sensor} \in \mathbb{R}^9$ denote the continuous 9-dimensional ambient telemetry feature vector, and let $\mathcal{X}_{vision} \in \mathbb{R}^{H \times W \times 3}$ denote an incoming RGB video frame. The ground truth fire state is defined as a binary random variable:

$$Y \in \{0, 1\} \quad \text{where } 0 = \text{Safe (Normal)}, 1 = \text{Active Fire/Smoke Emergency}$$

The primary objective of AFAS is to estimate the posterior probability of a fire event given multimodal observations:

$$P(Y = 1 \mid \mathcal{X}_{sensor}, \mathcal{X}_{vision})$$

And conditionally map this belief state to an optimal occupant-specific actuation vector $\mathbf{a} \in \{0, 1\}^3$ (comprising Visual Strobes, Tactile Bed Shakers, and Auditory Voice Beacons) and an egress path $\mathcal{P}^*$ across a building topology graph $G = (V, E)$.

### 1.2 Asymmetric Cost-Utility Loss Function
In safety-critical disaster detection, the cost of a **False Negative** ($C_{FN}$, failing to alert occupants of an active fire, resulting in loss of life) is infinitely greater than the cost of a **False Positive** ($C_{FP}$, a nuisance alarm triggered by kitchen cooking):

$$C_{FN} \gg C_{FP} \implies \text{Cost Ratio } \frac{C_{FN}}{C_{FP}} \ge 100$$

Under classical Bayesian decision theory, the optimal decision boundary $\theta^*$ that minimizes expected Bayes risk $\mathcal{R}$ is:

$$\theta^* = \frac{C_{FP}}{C_{FN} + C_{FP}}$$

For a typical life-safety system where $C_{FN} / C_{FP} = 100$, $\theta^* \approx 0.0099 \approx 0.01$. This mathematical reality governs the design of AFAS: **the fusion layer must never allow an occluded sensor or camera to suppress a positive detection on another channel**.

---

## 2. Sensor Branch: Tabular Ensemble Machine Learning

### 2.1 Physics & Chemical Sensing Principles of the 9 Channels
The 9 telemetry channels in the IoT Smoke Detection dataset ($N = 62,630$) capture complementary chemical and thermodynamic markers:

```
+-----------------------------------------------------------------------------------------------+
| Feature Channel      | Physical Phenomenon Measured          | Sensor Mechanism & Transduction |
+-----------------------------------------------------------------------------------------------+
| Barometric Pressure  | Sudden convective air currents & heat | Piezoresistive diaphragm bridge |
| TVOC                 | Volatile organic pyrolysis byproducts | Chemiresistive SnO2 thin film   |
| Ambient Humidity     | Rapid moisture evaporation near fire  | Capacitive polymer thin film    |
| Raw Ethanol Sensor   | Hydrocarbon combustion traces         | Chemiresistive semiconductor    |
| Raw H2 Sensor        | Early-stage smoldering pyrolysis gas  | Catalytic gas adsorption        |
| PM2.5 Concentration  | Fine airborne particulates (<2.5 um)  | Optical laser Mie scattering    |
| PM1.0 Concentration  | Ultra-fine soot condensation (<1.0 um)| Forward laser light scattering  |
| Temperature          | Convective & radiative heat release   | Negative Temp Coefficient (NTC) |
| eCO2 Concentration   | Carbon dioxide equivalent emissions   | Photoacoustic / Metal-oxide VOC |
+-----------------------------------------------------------------------------------------------+
```

*Key Insight:* Smoldering combustion releases gaseous VOCs and $\text{H}_2$ minutes before open flames break out. Ambient temperature is a **lagging indicator** (it rises only after combustion is well established).

---

### 2.2 Random Forest: Algorithmic Mechanics & Derivations
Random Forest constructs an ensemble of $B = 100$ decorrelated decision trees $\{T_1, T_2, \dots, T_B\}$.

#### 1. Splitting Criterion (Gini Impurity)
At each node $t$ containing $N_t$ samples across classes $k \in \{0, 1\}$, the Gini Impurity $I_G(t)$ is computed as:

$$I_G(t) = 1 - \sum_{k=0}^{1} p(k \mid t)^2 = 2 \cdot p(0 \mid t) \cdot p(1 \mid t)$$

The candidate split $\theta = (j, s)$ on feature $j$ at threshold $s$ partitions node $t$ into left child $t_L$ and right child $t_R$. The split that maximizes the Gini Impurity Reduction (Information Gain) is selected:

$$\Delta I_G(t, \theta) = I_G(t) - \left( \frac{N_{t_L}}{N_t} I_G(t_L) + \frac{N_{t_R}}{N_t} I_G(t_R) \right)$$

#### 2. Bagging & Variance Reduction
For an ensemble of $B$ trees with individual variance $\sigma^2$ and pairwise correlation $\rho$, the variance of the ensemble prediction is:

$$\text{Var}(\bar{T}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

By drawing bootstrap samples $\mathcal{D}_b \sim \mathcal{D}$ and randomly subsampling $m = \sqrt{p} = \sqrt{9} = 3$ features at each node, Random Forest drives $\rho \to 0$, reducing ensemble variance without increasing bias.

---

### 2.3 XGBoost: Second-Order Gradient Boosting Formulation
XGBoost minimizes a regularized objective function using a second-order Taylor expansion:

$$\mathcal{L}^{(t)} = \sum_{i=1}^n l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$

Where the tree complexity penalty is $\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$. Expanding via Taylor approximation around $\hat{y}_i^{(t-1)}$:

$$\tilde{\mathcal{L}}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$

Where $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ is the first-order gradient, and $h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ is the second-order Hessian. For binary logloss:

$$g_i = p_i - y_i, \quad h_i = p_i(1 - p_i)$$

For a leaf $j$ with sample index set $I_j = \{i \mid q(x_i) = j\}$, the optimal leaf weight $w_j^*$ and optimal split gain $\mathcal{G}$ are:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}, \quad \mathcal{G} = \frac{1}{2} \left[ \frac{\left(\sum_{i \in I_L} g_i\right)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{\left(\sum_{i \in I_R} g_i\right)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{\left(\sum_{i \in I} g_i\right)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$

---

### 2.4 Hyperparameter Profiles & Search Spaces
| Parameter Name | Model | Selected Value | Search Range Evaluated | Technical Justification |
|---|---|---|---|---|
| `n_estimators` | Random Forest | `100` | $50 - 500$ | Marginal variance reduction beyond 100 trees was $< 0.001\%$; 100 preserves $20\ \mu\text{s}$ latency. |
| `criterion` | Random Forest | `'gini'` | `['gini', 'entropy', 'log_loss']` | Gini computation avoids logarithmic evaluations, optimizing edge CPU cycles. |
| `max_features` | Random Forest | `'sqrt'` ($3$) | `['sqrt', 'log2', 0.5, None]` | $\sqrt{9} = 3$ maximizes decorrelation ($\rho$) between trees. |
| `min_samples_split` | Random Forest | `2` | $2 - 10$ | Pure leaf isolation for acute chemical spike detection. |
| `learning_rate` ($\eta$) | XGBoost | `0.1` | $0.01 - 0.3$ | Shrinkage factor weighting tree additions to prevent overfitting. |
| `max_depth` | XGBoost | `6` | $3 - 10$ | Limits tree interactions to 6th order feature combinations. |
| `reg_lambda` ($\lambda$) | XGBoost | `1.0` | $0.0 - 10.0$ | $L_2$ leaf regularization to prevent extreme leaf weights. |

---

## 3. Vision Branch: Deep Convolutional & Decoupled Object Detection

### 3.1 YOLOv8n Architecture: Layer-by-Layer Breakdown
YOLOv8n consists of **130 layers, 3,011,238 parameters, and 8.2 GFLOPs**.

```
                   INPUT: RGB Image (640 x 480 x 3)
                                  │
                                  ▼
[BACKBONE: CSPDarknet53]
├─ Conv 3x3, Stride 2 (P1: 320x240x16)
├─ Conv 3x3, Stride 2 (P2: 160x120x32) + C2f(n=1)
├─ Conv 3x3, Stride 2 (P3: 80x60x64)   + C2f(n=2)  ──> P3 (Small smoke plumes)
├─ Conv 3x3, Stride 2 (P4: 40x30x128)  + C2f(n=2)  ──> P4 (Medium flames)
├─ Conv 3x3, Stride 2 (P5: 20x15x256)  + C2f(n=1)  ──> P5 (Large fire outbreaks)
└─ SPPF (Spatial Pyramid Pooling Fast: 5x5 maxpool cascade)
                                  │
                                  ▼
[NECK: PANet Feature Pyramid]
├─ Top-Down Upsampling & Concatenation (P5 -> P4 -> P3)
└─ Bottom-Up Downsampling & Concatenation (P3 -> P4 -> P5)
                                  │
                                  ▼
[HEAD: Decoupled Anchor-Free Detection]
├─ Classification Branch (BCE Loss)   ──> Class Confidences [P(smoke), P(fire)]
└─ Regression Branch (CIoU + DFL Loss) ──> Bounding Box Coordinates [x, y, w, h]
```

* **C2f Module (Cross Stage Partial with 2 Convolutions):** Replaces legacy C3 blocks. Splits the input tensor into two branches; one branch passes through a series of Darknet bottleneck blocks with residual skips, while the other is concatenated directly, enhancing gradient propagation without adding FLOPs.
* **SiLU Activation:** Used in all convolutional layers:
  $$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

---

### 3.2 Loss Function Formulations
YOLOv8 trains without predefined anchor boxes using a multi-task loss:

$$\mathcal{L}_{total} = \lambda_{cls} \mathcal{L}_{BCE} + \lambda_{box} \mathcal{L}_{CIoU} + \lambda_{dfl} \mathcal{L}_{DFL}$$

#### 1. Binary Cross-Entropy Classification Loss ($\mathcal{L}_{BCE}$)
$$\mathcal{L}_{BCE} = -\sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

#### 2. Complete Intersection over Union Bounding Box Loss ($\mathcal{L}_{CIoU}$)
CIoU incorporates overlap area, central point distance, and aspect ratio consistency:

$$\mathcal{L}_{CIoU} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

Where:
* $\rho(b, b^{gt})$ is the Euclidean distance between predicted center $b$ and ground truth center $b^{gt}$.
* $c$ is the diagonal length of the smallest enclosing bounding box.
* $v = \frac{4}{\pi^2} \left( \arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h} \right)^2$ measures aspect ratio disparity.
* $\alpha = \frac{v}{(1 - \text{IoU}) + v}$ is a dynamic weighting factor.

#### 3. Distribution Focal Loss ($\mathcal{L}_{DFL}$)
Rather than predicting a single deterministic coordinate, YOLOv8 models bounding box edges as a continuous probability distribution across discrete bins, allowing the network to capture spatial uncertainty in diffuse, irregular smoke edges.

---

### 3.3 MobileNetV2 Architecture & Comparative Baseline
* **Parameters:** 2,225,153 parameters.
* **Core Innovation:** **Inverted Residuals with Linear Bottlenecks**. Expands the low-dimensional input to high-dimensional space ($6\times$ expansion factor) via $1 \times 1$ pointwise conv, applies $3 \times 3$ Depthwise Separable Convolution, and projects back with a linear bottleneck (no non-linearity) to preserve information manifolds.
* **Why YOLOv8n was Chosen:** MobileNetV2 outputs only a global scalar probability $P(fire)$. It cannot identify *where* the fire is located, how many distinct fire sources exist, or detect smoke plumes before open flames appear.

---

### 3.4 FLIR Thermal Vision Simulation Physics
FLIR infrared thermal sensors detect long-wavelength infrared radiation (LWIR: $8 - 14\ \mu\text{m}$). Blackbody spectral radiance is governed by Planck's Law:

$$B(\lambda, T) = \frac{2hc^2}{\lambda^5 \left(e^{\frac{hc}{\lambda k_B T}} - 1\right)}$$

Total radiant emittance scales with the fourth power of absolute temperature (Stefan-Boltzmann Law: $j^* = \sigma T^4$). In AFAS:
1. Grayscale luminance conversion maps relative thermal emission:
   $$Y = 0.299R + 0.587G + 0.114B$$
2. Saturated heat pixels are isolated via thresholding $I(x, y) > 245$.
3. Area coverage ratio $R_{bright} = \frac{\sum [I(x, y) > 245]}{H \times W} > 0.005$ triggers high-confidence fire recognition ($P = 0.99$) even in total optical darkness (Scenario SC-6).

---

## 4. Multimodal Fusion Engine & Calibration Theory

### 4.1 Ambient Baseline Removal Calibration
In standard non-fire rooms, normal ambient gas variations cause raw sensor models to output baseline noise of $0.30 - 0.35$. AFAS applies linear baseline rectification:

$$P_{sensor\_calibrated} = \max\left(0.0,\ \frac{P_{sensor} - 0.35}{0.65}\right)$$

* When $P_{sensor} = 0.35 \implies P_{sensor\_calibrated} = 0.00$.
* When $P_{sensor} = 1.00 \implies P_{sensor\_calibrated} = 1.00$.

### 4.2 Asymmetric Safety-Critical OR-Gate vs. Weighted Averaging
Consider a common emergency scenario where a fire starts behind a cabinet (camera occluded: $P_{vision} = 0.00$, sensors detect high smoke: $P_{sensor\_cal} = 0.90$).

* **Linear Weighted Average:**
  $$P_{fused} = 0.5 \cdot (0.00) + 0.5 \cdot (0.90) = 0.45 \quad (\text{BELOW ALARM THRESHOLD 0.50 } \implies \textbf{FATAL FAILURE})$$
* **AFAS Asymmetric OR-Gate:**
  $$P_{fused} = \max(P_{vision}, P_{sensor\_cal}) = \max(0.00, 0.90) = 0.90 \quad (\textbf{HIGH SEVERITY ALARM TRIGGERED})$$

### 4.3 Domain-Grounded False Positive Suppression Heuristics
* If $T < 35^\circ\text{C}$ and $\text{TVOC} < 1500\text{ ppb} \implies P_{sensor} \leftarrow P_{sensor} \times 0.10$.
* If $T > 50^\circ\text{C}$ or $\text{TVOC} > 2000\text{ ppb} \implies P_{sensor} \leftarrow \max(P_{sensor}, 0.90)$.

---

## 5. Graph-Theoretic Accessible Egress Planning

### 5.1 Graph Representation & Adjacency Modeling
AFAS models the building floor plan as an undirected, weighted graph $G = (V, E, W)$:
* **Vertices $V$:** Architectural waypoints: `{"Room 101", "Corridor A", "Corridor B", "Stairwell", "Main Exit", "Accessible Exit"}`.
* **Edges $E$:** Navigable corridors connecting waypoints.
* **Attribute Mask:** Each edge $e \in E$ carries a boolean attribute `accessible`:
  * $(Corridor\ A, Stairwell) \to \text{accessible} = \text{False}$ (Stairway obstruction).
  * $(Corridor\ B, Accessible\ Exit) \to \text{accessible} = \text{True}$ (Ramp-equipped egress).

### 5.2 Dijkstra Shortest-Path Algorithm & Edge Pruning
For occupants registered with a mobility disability profile (`mobility`), AFAS induces an accessible subgraph $G_{acc} = (V, E_{acc})$ where $E_{acc} = \{e \in E \mid e.\text{accessible} = \text{True}\}$.

The optimal path $\mathcal{P}^* = \arg\min_{\mathcal{P}} \sum_{e \in \mathcal{P}} w(e)$ is solved using a min-priority queue (Dijkstra's Algorithm) with computational complexity $\mathcal{O}(|E| + |V| \log |V|)$.

```
[ Room 101 ] ──(Accessible)──> [ Corridor A ]
                                      │
            ┌─────────────────────────┴────────────────────────┐
            ▼                                                  ▼
     [ Corridor B ]                                     [ Stairwell ]
            │                                           (SUPPRESSED:
            ▼ (Ramp)                                    NOT ACCESSIBLE)
   [ ACCESSIBLE EXIT ]                                         │
   (SAFE WHEELCHAIR EGRESS)                                    ▼
                                                          [ Main Exit ]
```

---

## 6. Empirical Results Showcase & Visual Analytics

### 6.1 Gini Impurity Feature Importance Distribution
Generated by [`ml/generate_visualizations.py`](file:///f:/Coding%20Workspace/AFAS/ml/generate_visualizations.py):

![Sensor Feature Importances](file:///f:/Coding%20Workspace/AFAS/static/charts/sensor_feature_importances.png)

```
Top Contributing Channels:
1. Barometric Pressure  : 26.98%  (Immediate pressure drop from thermal convection)
2. TVOC Organics        : 26.77%  (Volatile chemical markers released during pyrolysis)
3. Ambient Humidity     : 12.54%  (Rapid localized moisture drop)
4. Raw Ethanol Sensor   :  9.72%  (Combustion hydrocarbon traces)
5. Raw H2 Sensor        :  6.65%  (Early-stage smoldering combustion gas)
```

---

### 6.2 Confusion Matrices (Sensors, Clean Vision, Stress-Test Vision)
Comprehensive 3-panel confusion matrix comparison across all test benchmarks:

![Confusion Matrices All](file:///f:/Coding%20Workspace/AFAS/static/charts/confusion_matrices_all.png)

* **Panel A (Sensor Random Forest, $N = 12,526$):** $3,575$ True Negatives, $8,951$ True Positives, **0 False Positives, 0 False Negatives** ($100.0\%$ accuracy).
* **Panel B (YOLOv8n Clean Benchmark, $N = 999$):** $223$ True Negatives, $739$ True Positives, $21$ False Positives, $16$ False Negatives ($96.30\%$ accuracy).
* **Panel C (YOLOv8n Expanded Stress Benchmark, $N = 2,249$):** $636$ True Negatives, $1,507$ True Positives, $58$ False Positives, $48$ False Negatives ($95.29\%$ accuracy).

---

### 6.3 Environmental Robustness Breakdown Across 8 Conditions
Performance across severe environmental degradation ($N = 2,249$ images):

![Environmental Stress Breakdown](file:///f:/Coding%20Workspace/AFAS/static/charts/environmental_stress_breakdown.png)

| Condition / Test Category | Sample Count ($N$) | Accuracy (%) | Performance Characteristic |
|---|:---:|:---:|---|
| **Dense Smoke Haze Fire** | 300 | **98.00%** | Multiscale PANet captures diffuse smoke edges |
| **Original Clean Fire** | 755 | **97.88%** | Sharp open-flame detection |
| **Motion Blur & Alarm Noise Fire** | 200 | **96.50%** | Robust to optical vibrations |
| **Low-Light / Night Fire** | 300 | **93.67%** | Minor sensitivity loss on faint smoke plumes in darkness |
| **Sunset / Warm Light Distractor (Safe)** | 200 | **93.50%** | Rejects warm tungsten/amber lighting ($93.5\%$ specificity) |
| **Night Safe Room** | 150 | **92.00%** | Suppresses dark shadows and safe room clutter |
| **Original Clean Safe Room** | 244 | **91.39%** | Standard clean non-fire baseline |
| **Steam / Fog Safe Room** | 100 | **88.00%** | 12% false triggers mitigated by multimodal sensor fusion |

---

### 6.4 2D Multimodal Decision Surface & Scenario Validation
Contour mapping illustrating the calibrated decision surface and scenario placements:

![Multimodal Fusion Decision Surface](file:///f:/Coding%20Workspace/AFAS/static/charts/multimodal_fusion_surface.png)

* **Yellow Dashed Boundary:** Medium Severity Fire cutoff ($P_{fused} \ge 0.50$).
* **Red Solid Boundary:** High Severity Emergency cutoff ($P_{fused} \ge 0.80$).
* Validated edge scenarios SC-1 through SC-6 map into appropriate safety quadrants.

---

### 6.5 ROC-AUC Characteristics
Receiver Operating Characteristic curves demonstrating discriminative power of the tabular sensor classifiers:

![ROC Curves Comparison](file:///f:/Coding%20Workspace/AFAS/static/charts/roc_curves_comparison.png)

Both Random Forest and XGBoost achieve an empirical **Area Under Curve (AUC) of 1.0000**, confirming distinct separation between safe room ambient gas baselines and active fire chemical emissions.

---

### 6.6 Topological Graph Egress Visualization
Network layout showing dynamic pathfinding for wheelchair occupants:

![Accessible Routing Graph](file:///f:/Coding%20Workspace/AFAS/static/charts/accessible_routing_graph.png)

The pathfinding module identifies the non-accessible stairwell edge, prunes it from the graph, and navigates wheelchair occupants safely:
$$\text{Room 101} \longrightarrow \text{Corridor A} \longrightarrow \text{Corridor B} \longrightarrow \text{Accessible Exit (Ramp)}$$

---

## 7. Hardware Requirements & Edge Deployment Specifications

| Deployment Tier | Target Hardware | Precision / Format | Expected Vision Latency | Expected FPS | Viability |
|---|---|---|:---:|:---:|---|
| **Tier 1 (Current Local Demo)** | Intel Core i5/i7 (CPU only) | FP32 | $40.5\text{ ms}$ | $\approx 24.7\text{ FPS}$ | Fully operational on host PC |
| **Tier 2 (Embedded Edge Host)** | Raspberry Pi 5 (4GB RAM) | ONNX Runtime (FP32) | $85.0\text{ ms}$ | $\approx 11.8\text{ FPS}$ | Viable for 1 Hz polling cycle |
| **Tier 3 (Edge AI Accelerator)** | Raspberry Pi + Google Coral TPU | INT8 Quantized | $14.2\text{ ms}$ | $\approx 70.4\text{ FPS}$ | Ultra-fast low-power appliance |
| **Tier 4 (Industrial Edge Gateway)**| NVIDIA Jetson Orin Nano (8GB) | TensorRT (FP16) | $4.8\text{ ms}$ | $> 200\text{ FPS}$ | Multi-camera institutional hub |

---

## 8. Comprehensive Viva Voce & Technical Defense Guide

#### Q1: What is the formal machine learning problem formulation of AFAS?
*Answer:* AFAS is a multimodal binary classification and cost-sensitive decision problem. It estimates posterior probability $P(Y=1 \mid \mathcal{X}_{sensor}, \mathcal{X}_{vision})$, where $\mathcal{X}_{sensor} \in \mathbb{R}^9$ is continuous tabular telemetry and $\mathcal{X}_{vision} \in \mathbb{R}^{H \times W \times 3}$ is an optical video frame, followed by cost-utility mapping to an actuator vector $\mathbf{a} \in \{0, 1\}^3$ and shortest-path egress routing on a graph $G=(V, E)$.

#### Q2: Why did you choose Random Forest over a Deep Multilayer Perceptron (MLP) for sensor data?
*Answer:* Tabular data exhibits heterogeneous features with non-smooth decision boundaries (e.g., discrete chemical concentration thresholds). Extensive empirical benchmarks (Grinsztajn et al., NeurIPS 2022) prove that tree-based ensembles (Random Forest / XGBoost) consistently outperform deep neural networks on tabular datasets because decision trees handle coordinate-aligned step-function splits naturally, are invariant to feature scaling, avoid gradient vanishing, and execute in $20\ \mu\text{s}$ compared to milliseconds for neural forward passes.

#### Q3: Why is Barometric Pressure the top feature in your Random Forest model?
*Answer:* As combustion initiates, localized intense thermal heating causes rapid air expansion and convective updrafts, creating transient micro-barometric pressure fluctuations. When sealed rooms experience flame ignition, pressure drops and drafts propagate faster than gaseous diffusion, making barometric pressure a key early indicator alongside TVOC.

#### Q4: Why is YOLOv8n superior to MobileNetV2 for this project?
*Answer:* MobileNetV2 is an image-level classifier that outputs a single scalar probability. It suffers from three major flaws: (1) no spatial localization (cannot locate *where* the fire is), (2) high false-positive susceptibility to sunsets, red lights, or warm textures, and (3) inability to differentiate smoke plumes from flames. YOLOv8n uses an anchor-free decoupled head to produce bounding boxes for both `smoke` (early smoldering warning) and `fire` (active flame outbreak).

#### Q5: Explain the mathematical purpose of the Baseline Removal Calibration formula.
*Answer:* In real-world environments, normal ambient air contains background traces of VOCs and humidity, causing the raw sensor model to output a baseline probability around $0.35$. The calibration formula $P_{cal} = \max(0, (P - 0.35) / 0.65)$ maps this ambient noise plateau to exactly $0.0$, ensuring that normal room air never generates residual alarm probabilities.

#### Q6: Why did you use an Asymmetric OR-Gate instead of a 50/50 weighted average in the Fusion Layer?
*Answer:* In life safety, False Negatives carry catastrophic consequences ($C_{FN} \gg C_{FP}$). If a fire starts behind a partition where the camera cannot see it ($P_{vision} = 0.00$), a $50/50$ weighted average would halve an acute sensor probability of $0.90$ down to $0.45$, falling below the $0.50$ alarm cutoff and failing to alert sleeping occupants. An asymmetric OR-gate ($\max(P_{vision}, P_{sensor\_cal})$) ensures that confirmed detection on *either* sensory pathway immediately triggers evacuation.

#### Q7: How does AFAS mitigate false alarms caused by kitchen cooking steam?
*Answer:* AFAS employs dual-stage validation: (1) In the vision branch, steam produces an $88\%$ specificity (12% false alarms) because white steam resembles smoke. (2) However, in the fusion layer, domain heuristics check sensor telemetry: kitchen steam causes high humidity but leaves ambient room temperature $< 35^\circ\text{C}$ and TVOC $< 1500\text{ ppb}$. The heuristic suppresses sensor probability by $90\%$, and the fusion gate prevents alarm activation (Scenario SC-2).

#### Q8: What loss functions are used in training YOLOv8?
*Answer:* YOLOv8 uses a multi-task loss combining: (1) Binary Cross-Entropy ($\mathcal{L}_{BCE}$) for classification, (2) Complete Intersection over Union ($\mathcal{L}_{CIoU}$) for bounding box distance, scale, and aspect ratio alignment, and (3) Distribution Focal Loss ($\mathcal{L}_{DFL}$) for modeling box boundaries as probability distributions to handle diffuse, irregular smoke edges.

#### Q9: What is the significance of the C2f module in YOLOv8's backbone?
*Answer:* The C2f (Cross-Stage Partial with 2 Convolutions) module splits the input feature tensor into two paths: one path bypasses the computational blocks, while the other passes through Darknet bottleneck layers with residual skips before concatenation. This design maximizes gradient flow pathways during backpropagation and enriches multiscale feature representation while reducing computational bottlenecks.

#### Q10: How does the FLIR thermal simulation mode work mathematically?
*Answer:* Thermal cameras capture radiative emittance proportional to $\sigma T^4$ (Stefan-Boltzmann law). In our simulation, optical frames are transformed to grayscale luminance $Y = 0.299R + 0.587G + 0.114B$. Pixels with saturation intensity $I(x, y) > 245$ (top 4% luminance) are isolated. If the area ratio $R_{bright} = \frac{\sum [I > 245]}{H \times W} > 0.005$, confidence is assigned as $0.99$, enabling fire detection in zero-lux optical darkness.

#### Q11: How does the accessible route guidance algorithm work?
*Answer:* The building topology is modeled as an undirected graph $G=(V, E)$ using NetworkX. Waypoints are nodes, and hallways are edges. When an occupant has a registered `mobility` profile (wheelchair user), edges with `accessible=False` (stairways) are pruned to generate an accessible subgraph $G_{acc}$. Dijkstra's algorithm then computes the shortest path to an accessible ramp exit.

#### Q12: Why did MobileNetV2 show inverted predictions when first evaluated on the Kaggle FIRE dataset?
*Answer:* PyTorch's `datasets.ImageFolder` assigns integer class targets in alphabetical order of folder names: `fire_images` was assigned target `0`, and `non_fire_images` was assigned target `1`. Standard evaluation scripts assume class `1` is the positive class. When properly decoded (class $0 = \text{Fire}$), MobileNetV2 achieved $98.40\%$ accuracy.

#### Q13: What happens when the camera stream is completely lost or disconnected?
*Answer:* In [`ml/fusion.py`](file:///f:/Coding%20Workspace/AFAS/ml/fusion.py#L16-L18), the fusion layer detects `img_prob is None`. Rather than defaulting to zero or failing, it transitions to single-modality sensor inference, routing raw calibrated sensor probability directly to the decision engine.

#### Q14: How does the system adapt its alerts for a deaf occupant?
*Answer:* In [`ml/alert_engine.py`](file:///f:/Coding%20Workspace/AFAS/ml/alert_engine.py#L28-L32), the occupant profile `deaf` disables auditory sirens (which are useless and consume battery/power) and activates high-frequency visual strobe animations and high-amplitude bed shaker vibration pulses, paired with directional visual guidance.

#### Q15: How does the system adapt its alerts for a blind occupant?
*Answer:* In [`ml/alert_engine.py`](file:///f:/Coding%20Workspace/AFAS/ml/alert_engine.py#L33-L36), visual strobes are disabled, and auditory sirens and tactile bed shakers are activated, paired with voice prompts guiding the occupant toward auditory beacons along the egress route.

#### Q16: What is the inference latency of the pipeline, and can it run in real time?
*Answer:* Yes. On a standard CPU, YOLOv8n executes in $40.55\text{ ms}$ ($\approx 24.7\text{ FPS}$), the Random Forest sensor model executes in $20.41\ \mu\text{s}$, and the fusion and routing engines execute in $< 1\text{ ms}$. Total pipeline latency is under $45\text{ ms}$, exceeding the requirements for real-time 1 Hz safety monitoring.

#### Q17: What is Gini Impurity, and how does it differ from Entropy?
*Answer:* Gini Impurity measures the probability of misclassifying a randomly chosen element if it were randomly labeled according to the class distribution: $I_G = 1 - \sum p_i^2$. Entropy measures average information content: $H = -\sum p_i \log_2(p_i)$. Both produce nearly identical tree splits, but Gini avoids logarithmic computations, executing faster on edge hardware.

#### Q18: What is the purpose of Data Augmentation in your expanded testing?
*Answer:* Training and evaluating only on clear Kaggle photos introduces severe distribution shift when deploying in real buildings. Our expanded benchmark generated 1,250 augmented test cases (low-light power outages, dense smoke contrast loss, alarm motion vibrations, and warm sunset distractors) to demonstrate that the model maintains $> 95\%$ accuracy under harsh real-world conditions.

#### Q19: Can you upgrade to an even higher-level model if compute permits?
*Answer:* Yes. If discrete edge GPUs (e.g. NVIDIA Jetson Orin) are available, we can upgrade the vision branch to **YOLOv8s** (11.2M params) or **RT-DETR** (Real-Time Detection Transformer), which completely eliminates Non-Maximum Suppression (NMS) and uses cross-attention to resolve dense overlapping smoke clouds. For the sensor branch, we can deploy a 1D-CNN + BiLSTM to model rate-of-rise temporal sequences $\frac{d(\text{TVOC})}{dt}$.

#### Q20: What are the primary ethical and practical contributions of this project?
*Answer:* Prior fire safety systems treat alert delivery as homogeneous (one siren for all) and evacuation routing as accessibility-neutral (relying on stairs). AFAS bridges the gap between AI fire perception and assistive technology by providing personalized multi-modal alerts and stair-free egress routing, directly mitigating disproportionate fire casualties among people with disabilities.

---

*Manual compiled for PCCoE AIML Termwork Submission & Evaluation.*
