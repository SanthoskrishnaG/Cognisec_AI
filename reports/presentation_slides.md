# Technical Presentation: Cognisec AI — Network Behaviour Fingerprinting
## Cognisec AI: Network Behaviour Fingerprinting for Unseen Attack Detection — An Experimental Comparison of XGBoost and Variational Autoencoders
### Tagline: Intelligent Network Behaviour Analysis for Unseen Attack Detection

---

### Slide 1: Title, Branding & Research Motivation
- **Product Name:** Cognisec AI
- **Tagline:** Intelligent Network Behaviour Analysis for Unseen Attack Detection
- **Full Academic Title:** Cognisec AI: Network Behaviour Fingerprinting for Unseen Attack Detection — An Experimental Comparison of XGBoost and Variational Autoencoders
- **Research Question:** Can a deep learning model trained on normal network behaviour detect attack families excluded from supervised training, and how does its performance compare with XGBoost?
- **Dataset:** Authentic CICIDS2017 (Canadian Institute for Cybersecurity)
- **Models Evaluated:** Exactly Two — XGBoost (Supervised) vs. PyTorch Tabular VAE (Benign Anomaly Detection)

---

### Slide 2: Two Competing Operational Paradigms
- **XGBoost (Supervised Classifier):**
  - Trained on labeled benign flows and known attack families.
  - Learns explicit discriminative boundaries.
  - Known strength: Extremely high recall and low false alarms on known attacks.
  - Potential weakness: Suspected blind spots on novel/unseen threat families.
- **Variational Autoencoder (Normal-Behaviour Modeller):**
  - Trained **exclusively on benign network flows** to learn the normal traffic manifold.
  - Generates reconstruction anomaly scores: $\text{Score} = \frac{1}{D} \sum (x_i - \hat{x}_i)^2$.
  - Decision threshold calibrated on benign validation data (1.0% target false-alarm rate).
  - Known strength: No historical attack signatures required.
  - Potential weakness: Sensitive to benign variability; high false-alarm risk.

---

### Slide 3: Methodological Integrity & Leakage Prevention
- **Socket Identifier Stripping:** Removed `Flow ID`, `Source IP`, `Destination IP`, `Timestamp`, and ephemeral ports.
- **Dual Representation Pipeline:** Fitted scalers and median imputations exclusively on training data.
- **Zero-Contamination Partitioning:**
  - `Train (60%)`: Benign + Known Attacks (`DoS`, `PortScan`, `Brute_Force`).
  - `Validation (20%)`: Threshold tuning and early stopping.
  - `Test (20%)`: Benign + Known Attacks + **100% of Held-Out Families** (`DDoS`, `Web_Attack`, `Bot`, `Infiltration`).

---

### Slide 4: Experiment A — Known Attack Detection Benchmark
- **Evaluation:** Seen attack families evaluated against unseen test benign flows.
- **XGBoost Performance:**
  - PR-AUC: **0.9999** | Recall: **99.99%** | False Alarm Rate: **0.09%**
  - Latency: **0.0005 ms/flow** (1.96M flows/sec on CPU)
- **VAE Performance:**
  - PR-AUC: **0.5974** | Recall: **0.93%** | False Alarm Rate: **1.16%**
  - Latency: **0.0059 ms/flow** (169K flows/sec on CPU)
- **Statistical Significance:** Paired difference $\Delta F1 = 0.9814$ (95% CI: [0.9801, 0.9827], $p < 0.001$).

---

### Slide 5: Experiment B — Unseen Attack-Family Detection (Zero-Day)
- **Evaluation:** Performance on attack families completely excluded from training.
- **Disaggregated Results by Family:**
  - `DDoS (128,027 flows)`: XGBoost Recall = **63.01%** | VAE Recall = **9.38%**
  - `Web Attack (2,180 flows)`: XGBoost Recall = **90.23%** | VAE Recall = **0.00%**
  - `Infiltration (36 flows)`: XGBoost Recall = **0.00%** | VAE Recall = **16.67%**
  - `Botnet (1,966 flows)`: XGBoost Recall = **0.00%** | VAE Recall = **0.00%**
- **Overall Zero-Day Recall:** XGBoost = **62.50%** vs. VAE = **9.09%**.

---

### Slide 6: Experiment C — False-Alarm vs. Recall Trade-Offs
- **The Operational Dilemma:**
  - In enterprise networks, a 1% false alarm rate equals thousands of false alarms per hour.
  - Enforcing a 0.1% false-alarm budget restricts VAE recall to **1.57%**.
  - Relaxing false-alarm budget to 10% only increases VAE recall to **9.26%**.
- **Root Cause:** High-variance benign traffic creates an anomaly score overlap that compresses decision margins.

---

### Slide 7: Experiment D & F — Robustness & Complementarity
- **Distribution-Shift Stress Testing (Exp D):**
  - Scaling packet length features by $0.5\times$ to $2.0\times$ causes minimal degradation in XGBoost PR-AUC (0.9991 $\to$ 0.9964).
  - VAE PR-AUC steadily degrades (0.8032 $\to$ 0.7861).
- **Complementarity Breakdown (Exp F):**
  - 12,005 attacks detected by **both models** (9.08%).
  - 70,630 attacks detected by **XGBoost only** (53.42%).
  - 7 attacks detected by **VAE only** (0.01% — stealthy Infiltration flows).
  - 49,567 attacks **missed by both** (37.49% — low-rate botnets and slow attacks).

---

### Slide 8: Conclusions & Takeaways
1. **Supervised Generalisation is Underestimated:** XGBoost generalized to unseen DDoS and Web Attacks because flow kinematics (window sizes, packet length distributions) overlap with known DoS attacks.
2. **Benign Anomaly Detection has Limitations:** Constraining false alarms to practical levels severely restricts VAE recall on subtle threats.
3. **Defense-in-Depth Value:** The VAE detected infiltration flows that completely bypassed XGBoost, proving that anomaly detectors should be deployed as secondary tripwires for stealthy anomalies rather than primary replacements for supervised classifiers.
