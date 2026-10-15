# Cognisec AI — Comprehensive Research Methodology Notes

**Project Title:** Cognisec AI: Network Behaviour Fingerprinting for Unseen Attack Detection — An Experimental Comparison of XGBoost and Variational Autoencoders  
**Academic Focus:** Empirical Comparison of Supervised Decision Trees vs. Generative Latent Reconstruction for Network Intrusion Detection  
**Benchmark:** Canadian Institute for Cybersecurity CICIDS2017 Benchmark Dataset  

---

## 1. Research Objective & Core Theoretical Framework

The primary research objective is to empirically evaluate and contrast two fundamentally distinct paradigms for network flow intrusion detection:
1. **Supervised Binary Classification (XGBoost):** Optimizes discriminative decision boundaries using labeled historical traffic vectors of known attacks and benign communication.
2. **Unsupervised Normal-Behaviour Modelling (PyTorch Tabular VAE):** Learns the probabilistic continuous manifold of purely benign network traffic. Detection is framed as reconstruction divergence: anomalous flows yield high reconstruction error ($\mathcal{L}_{\text{recon}}$) under the generative hypothesis that unseen malicious patterns deviate from legitimate traffic semantics.

### The Research Question: Novel Zero-Day vs. Experimentally Held-Out Families
A central conceptual contribution of this methodology is strictly distinguishing **experimentally held-out attack families** from **truly novel real-world zero-day attacks**:
- In this research, an *experimentally held-out family* (`DDoS`, `Web_Attack`, `Bot`, `Infiltration`) is completely excluded from the training split, hyperparameter tuning, and threshold calibration.
- However, because all traffic originates from the same synthetic testbed environment (CICIDS2017), underlying network topologies, victim subnet ranges, and background operating system behaviors remain shared.
- Therefore, out-of-distribution performance on held-out families evaluates cross-attack structural transferability, but must not be overstated as proof of universal zero-day immunity.

---

## 2. Dataset Provenance and Preprocessing Protocol

### 2.1 CICIDS2017 Provenance
The benchmark utilizes authentic network flow data from the Canadian Institute for Cybersecurity (CIC), University of New Brunswick (UNB):
- **Citation:** Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization." *Proceedings of the International Conference on Information Systems Security and Privacy (ICISSP)*, 108–116. DOI: [https://doi.org/10.5220/0006639801080116](https://doi.org/10.5220/0006639801080116)
- **Official Repository:** [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Raw CSV File Inventory in Project:**
  1. `Friday-WorkingHours-Afternoon-DDos.csv` (57.06 MB)
  2. `Friday-WorkingHours-Afternoon-PortScan.csv` (52.21 MB)
  3. `Friday-WorkingHours-Morning.csv` (9.01 MB)
  4. `Monday-WorkingHours.csv` (9.03 MB)
  5. `Thursday-WorkingHours-Afternoon-Infilteration.csv` (7.98 MB)
  6. `Thursday-WorkingHours-Morning-WebAttacks.csv` (9.09 MB)
  7. `Tuesday-WorkingHours.csv` (13.23 MB)
  8. `Wednesday-workingHours.csv` (105.30 MB)
- **Total Ingested Records:** 757,646 raw flow records across all 8 capture files.

### 2.2 Leakage Prevention & Feature Selection
To guarantee that models learn genuine traffic kinematics rather than memorizing laboratory artifacts, the pipeline enforces:
1. **Unconditional Socket Dropping:** All socket-level identifiers (`Flow ID`, `Source IP`, `Destination IP`, `Timestamp`, `External IP`) were excised prior to any scaling or feature extraction.
2. **Infinite and Missing Value Imputation:** Network features with division-by-zero occurrences (`Flow Bytes/s`, `Flow Packets/s`) contain `+inf` and `-inf`. These were identified, coerced to NaN, and imputed with column medians computed **exclusively from the training partition**.
3. **Feature Variance Auditing:** Zero-variance constant features across benign traffic were dropped, yielding **exactly 70 clean numerical behavioral features**.
4. **Log-Variance Stabilization:** Heavy-tailed burst metrics were log1p-transformed prior to `RobustScaler` (median-centered, IQR-scaled) with hard clipping $[-5.0, 5.0]$ to prevent outlier divergence in the neural autoencoder.

---

## 3. Experimental Design and Data Partitioning

### 3.1 Strict Partition Isolation
Data splitting follows a 60% Train / 20% Validation / 20% Test stratified structure (`random_seed=42`) with **zero held-out leakage**:
- **Train (375,262 flows):** 120,000 Benign + 255,262 Known Attacks (`DoS`, `PortScan`, `Brute_Force`, `Heartbleed`).
- **Validation (125,087 flows):** 40,000 Benign + 85,087 Known Attacks. Used exclusively for XGBoost early stopping and VAE threshold calibration.
- **Test (257,297 physical flows):** 40,000 Benign + 85,088 Known Attacks + 132,209 Held-Out Attacks (`DDoS`, `Web_Attack`, `Bot`, `Infiltration`).

### 3.2 Sub-Benchmark Design
To prevent cross-contamination between known and unknown evaluations:
- **Experiment A (Known Attack Benchmark):** Evaluates models on 40,000 Benign + 85,088 Known Attacks ($N = 125,088$).
- **Experiment B (Held-Out Zero-Day Benchmark):** Evaluates models on the exact same 40,000 Benign + 132,209 Held-Out Attacks ($N = 172,209$).
- Total evaluation decisions executed: $125,088 + 172,209 = 297,297$.
- Unique physical test flows: $297,297 - 40,000 (\text{shared benign}) = 257,297$.

---

## 4. Model Architectures & Calibration Protocols

### 4.1 Supervised XGBoost Classifier
- **Implementation:** `xgboost.XGBClassifier` with histogram tree construction (`tree_method="hist"`).
- **Hyperparameters:** `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `objective="binary:logistic"`, `eval_metric="logloss"`, `random_seed=42`.
- **Early Stopping:** 15 validation rounds.
- **Threshold Calibration:** Scanned validation split across 81 operating thresholds $\tau \in [0.10, 0.90]$. Optimal F1 threshold identified as **$\tau = 0.52$**.
- **CPU Inference Speed:** 0.00048 ms/flow (~2.08 million flows/second).

### 4.2 Unsupervised PyTorch Tabular VAE
- **Architecture:** Symmetrical bottleneck neural network.
  - Encoder: Linear(70 $\to$ 64) $\to$ BatchNorm $\to$ LeakyReLU(0.2) $\to$ Dropout(0.1) $\to$ Linear(64 $\to$ 32) $\to$ Linear(32 $\to$ 16 for $\mu$ and $\log \sigma^2$).
  - Reparameterization: $z = \mu + \epsilon \odot \exp(0.5 \log \sigma^2)$, $\epsilon \sim \mathcal{N}(0, I_{16})$.
  - Decoder: Linear(16 $\to$ 32) $\to$ BatchNorm $\to$ LeakyReLU(0.2) $\to$ Linear(32 $\to$ 64) $\to$ Linear(64 $\to$ 70).
- **Loss Formulation:**
  $$\mathcal{L}(x, \hat{x}) = \frac{1}{D} \sum_{j=1}^D (x_j - \hat{x}_j)^2 + \beta \cdot D_{KL}\left(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, I)\right)$$
  where $\beta = 0.005$, balancing reconstruction fidelity against latent regularization.
- **Training Restriction:** Trained strictly on 120,000 benign training flows for 15 epochs with AdamW (`lr=0.001`, `weight_decay=1e-5`, `batch_size=512`).
- **Anomaly Score:** Mean squared feature reconstruction error:
  $$S(x) = \frac{1}{70} \sum_{j=1}^{70} (x_j - \hat{x}_j)^2$$
- **Operational Threshold Calibration:** Fixed strictly on benign validation scores to satisfy an operational target of **1.0% False Positive Rate** (99th percentile): **$\tau = 0.330610$**.
- **CPU Inference Speed:** 0.00399 ms/flow (~250,900 flows/second).

---

## 5. Metrics and Evaluation Definitions

Given extreme class imbalance and operational costs in enterprise security operations:
1. **Precision ($\frac{TP}{TP + FP}$):** Ratio of true attacks among raised alerts. Critical for preventing SOC alert fatigue.
2. **Recall / Detection Rate ($\frac{TP}{TP + FN}$):** Ratio of detected attacks among total malicious traffic.
3. **False Positive Rate ($\frac{FP}{FP + TN}$):** Fraction of legitimate benign traffic incorrectly flagged as malicious.
4. **F1-Score ($2 \cdot \frac{P \cdot R}{P + R}$):** Harmonic balance between precision and recall.
5. **PR-AUC (Precision-Recall Area Under Curve):** Primary metric for imbalanced evaluation; measures trade-off without being distorted by high true negative counts.
6. **ROC-AUC:** Measures discrimination across all possible decision boundaries.
7. **Inferential Significance:** Paired empirical bootstrap resampling ($B=1,000$) and McNemar's discordance test ($\chi^2$).
