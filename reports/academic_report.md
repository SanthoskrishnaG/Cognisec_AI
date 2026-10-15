# Network Behaviour Fingerprinting for Unseen Attack Detection: An Experimental Comparison of XGBoost and Variational Autoencoders

**Author:** Cybersecurity & Machine Learning Research Team  
**Dataset:** Authentic CICIDS2017 (Canadian Institute for Cybersecurity)  
**Academic Domain:** Network Intrusion Detection, Tabular Deep Learning, Anomaly Detection  

---

## 1. Abstract
Contemporary Network Intrusion Detection Systems (NIDS) are challenged by rapidly evolving zero-day attacks that evade signature-based detection. This study presents a rigorous experimental investigation comparing two foundational paradigms: supervised classification using **Extreme Gradient Boosting (XGBoost)** and semi-supervised normal-behaviour modelling using a **PyTorch Variational Autoencoder (VAE)**. Using authentic flow records from the **CICIDS2017** benchmark, we enforce strict leakage controls, time-aware partitioning, and held-out attack-family isolation (`DDoS`, `Web_Attack`, `Bot`, `Infiltration`). Empirical findings demonstrate that while XGBoost exhibits near-perfect discrimination on represented attack families (PR-AUC: 0.9999, Recall: 99.99%, FPR: 0.09%), it experiences selective degradation on unseen families (overall held-out recall: 62.50%). Conversely, the VAE—constrained to a 1.0% operational false-alarm budget on benign validation data—achieves an overall held-out recall of 9.09% but uniquely detects 16.67% of stealthy `Infiltration` attacks that completely evaded XGBoost (0.00% recall). Paired bootstrap hypothesis tests confirm statistical significance ($p < 0.001$). These results challenge the prevailing assumption that benign anomaly detectors universally outperform supervised models on zero-day attacks, highlighting targeted complementarity over universal superiority.

---

## 2. Introduction
Modern network infrastructures are subjected to increasingly sophisticated cyberattacks designed to mimic legitimate protocol transactions and evade static firewall filters. While supervised machine learning algorithms have achieved high benchmark accuracies on standard datasets, their capacity to generalize to novel attack families excluded from training remains contentious. Anomaly detection via Variational Autoencoders represents a promising alternative: by learning the probabilistic latent manifold of benign network flows, abnormal flows should yield high reconstruction errors without requiring historical attack signatures. This research experimentally examines both paradigms under identical, leakage-controlled conditions.

---

## 3. Problem Statement
Supervised intrusion detection models rely heavily on labeled historical attack flows, inducing inductive bias toward known threat vectors. In operational environments, novel threats (zero-day exploits, advanced persistent threats) emerge without available training signatures. Concurrently, benign-only anomaly detection models frequently suffer from excessive false-alarm rates and unpredictable sensitivity across diverse attack profiles. The research problem addresses whether a deep learning model trained exclusively on normal network behavior can reliably detect attack families excluded from supervised training, and how its detection capabilities, false-alarm rates, and computational overhead compare with supervised gradient boosting.

---

## 4. Research Aim and Objectives
**Central Research Aim:**  
Design, implement, and evaluate a reproducible network intrusion detection framework that empirically compares supervised classification (XGBoost) and normal-behaviour anomaly modelling (VAE) on known and unseen cyberattacks.

**Specific Research Objectives:**
1. Ingest authentic CICIDS2017 machine-learning flow records and conduct a rigorous data quality and identifier leakage audit.
2. Establish zero-leakage partitions isolating specific attack families (`DDoS`, `Web_Attack`, `Bot`, `Infiltration`) into the test partition.
3. Develop, optimize, and validate an XGBoost classifier with validation-based threshold optimization.
4. Implement and train a PyTorch Tabular VAE trained exclusively on benign network traffic with percentile-based threshold calibration on benign validation records.
5. Execute controlled benchmarks: Known Attack Detection (Exp A), Held-Out Attack Family Generalisation (Exp B), False-Alarm Sensitivity (Exp C), Distribution-Shift Robustness (Exp D), and Complementarity Disagreement Analysis (Exp F).
6. Perform statistical significance testing via empirical paired bootstrap confidence intervals.

---

## 5. Literature Review
The application of machine learning to network intrusion detection originated with classical algorithms including Random Forests, Support Vector Machines, and Gradient Boosted Decision Trees (Chen & Guestrin, 2016). Sharafaldin et al. (2018) introduced CICIDS2017 to rectify deficiencies in legacy datasets (KDD99, NSL-KDD), offering realistic background traffic and multi-day attack scenarios. 

However, recent critical appraisals (Arp et al., 2022; Engelen et al., 2021) revealed widespread methodological flaws in published NIDS literature, including feature leakage (e.g., models memorizing IP addresses or flow identifiers rather than traffic semantics), inappropriate random row splitting that contaminates train/test sets with correlated packet bursts, and threshold leakage where anomaly decision boundaries are tuned using test attack labels. Anomaly detection using autoencoders and VAEs (Kingma & Welling, 2013) has gained traction for tabular network data, yet rigorous out-of-sample comparisons against state-of-the-art gradient boosting remain sparse.

---

## 6. Research Gap and Motivation
Prior comparative studies often make one of three fatal methodological concessions:
1. Testing models on identical attack families present in training, creating an illusion of generalisation that collapses on unseen attacks.
2. Selecting anomaly thresholds by maximizing F1-score on the final test set, which violates operational causality and inflates reported anomaly performance.
3. Introducing third-party ensemble classifiers or hybridizing models instead of isolating the fundamental strengths and limitations of each paradigm.

This research directly addresses these gaps by evaluating both paradigms under strict zero-leakage protocols with pre-declared threshold calibration on benign validation data.

---

## 7. Dataset Description
The research uses authentic flow CSV records from the **CICIDS2017** benchmark, generated by the Canadian Institute for Cybersecurity:
- **Total Ingested Records:** 757,646 flow records across 8 daily capture files.
- **Benign Baseline:** 200,000 authentic benign flows sampled across all capture days.
- **Attack Records:** 557,646 authentic attack flows encompassing DoS (Hulk, GoldenEye, Slowloris, Slowhttptest), PortScan, Brute Force (FTP-Patator, SSH-Patator), DDoS, Web Attacks (Brute Force, XSS, SQL Injection), Botnets, and Infiltration.
- **Attack Family Distribution:**
  - `DoS`: 252,661 records
  - `BENIGN`: 200,000 records
  - `PortScan`: 158,930 records
  - `DDoS`: 128,027 records
  - `Brute_Force`: 13,835 records
  - `Web_Attack`: 2,180 records
  - `Bot`: 1,966 records
  - `Infiltration`: 36 records
  - `Heartbleed`: 11 records

---

## 8. Data Preprocessing and Leakage Prevention
To ensure rigorous research validity, the pipeline implements multi-stage data engineering:
1. **Identifier Leakage Stripping:** All socket identities (`Flow ID`, `Source IP`, `Destination IP`, `Timestamp`, `External IP`) were unconditionally removed to prevent models from learning network topology artifacts.
2. **Infinite and Missing Value Handling:** Features such as `Flow Bytes/s` and `Flow Packets/s` containing `inf` or `-inf` values were converted to NaN and imputed using median values computed exclusively from the training split.
3. **Dual Model Representation:**
   - **XGBoost:** Imputed unscaled tabular representation preserving original feature scales.
   - **VAE:** Imputed and standardized using `RobustScaler` (median-centered and interquartile-range scaled) fitted exclusively on the training partition.
4. **Feature Variance Auditing:** Zero-variance constant features were dropped, yielding **70 clean behavioral flow features**.

---

## 9. Methodology
The research is organized into a formal hypothesis-driven experimental pipeline:
- **Hypothesis 1 (H1):** XGBoost will achieve superior recall and lower false-alarm rates on attack families represented in training.
- **Hypothesis 2 (H2):** The VAE will detect zero-day attack families at non-zero rates when operating under a strict 1% benign false-alarm constraint.
- **Hypothesis 3 (H3):** The VAE will identify specific stealth attack instances that completely evade XGBoost.
- **Hypothesis 4 (H4):** Detection sensitivity varies drastically across attack families based on flow volumetric footprint.
- **Hypothesis 5 (H5):** XGBoost decision tree structures exhibit higher resilience to feature scale shifts than continuous latent reconstruction.

---

## 10. XGBoost Implementation
The supervised model utilizes `xgboost.XGBClassifier` with histogram-based tree construction (`tree_method="hist"`):
- Hyperparameters: `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `objective="binary:logistic"`, `eval_metric="logloss"`.
- Decision Threshold: Optimized strictly on the validation set to maximize F1-score, yielding an optimal operating threshold of **$\tau = 0.52$**.
- Training duration: **6.12 seconds** on CPU.

---

## 11. VAE Architecture and Training
The deep learning model is implemented in PyTorch (`TabularVAE`):
- **Encoder:** $70 \to 64 \to 32 \to 16$ (latent mean $\mu$ and log-variance $\log \sigma^2$) with Batch Normalization, LeakyReLU ($\alpha = 0.2$), and Dropout ($p = 0.1$).
- **Latent Space:** 16 dimensions with Gaussian reparameterization: $z = \mu + \epsilon \odot \exp(0.5 \log \sigma^2)$, $\epsilon \sim \mathcal{N}(0, I)$.
- **Decoder:** $16 \to 32 \to 64 \to 70$ with Batch Normalization and LeakyReLU.
- **Loss Objective:** $\mathcal{L} = \text{MSE Reconstruction Loss} + \beta \times D_{KL}$, where $\beta = 0.005$.
- **Training Protocol:** Trained exclusively on 120,000 benign flows with early stopping on benign validation loss.
- **Threshold Calibration:** Fixed at the 99th percentile of benign validation reconstruction scores, enforcing an operational target of **1.0% False Positive Rate** ($\tau = 6.06 \times 10^{12}$).

---

## 12. Experimental Design
The dataset was partitioned into:
- **Train (60% - 375,262 flows):** Benign flows + Known Attack Families (`DoS`, `PortScan`, `Brute_Force`).
- **Validation (20% - 125,087 flows):** Used exclusively for threshold selection and early stopping.
- **Test (20% - 257,297 flows):** Composed of unseen benign traffic, known attack families, and **100% of held-out attack families** (`DDoS`, `Web_Attack`, `Bot`, `Infiltration`).

Five distinct experiments were conducted:
- **Experiment A:** Known Attack Detection Benchmark.
- **Experiment B:** Held-Out (Zero-Day) Attack-Family Detection.
- **Experiment C:** False-Alarm and Threshold Sensitivity Analysis.
- **Experiment D:** Distribution-Shift Robustness (Packet size scaling from $0.5\times$ to $2.0\times$).
- **Experiment F:** Model Complementarity & Disagreement Analysis.

---

## 13. Evaluation Metrics
Models were assessed using standard classification and operational metrics:
- Precision, Recall, F1-Score, False Positive Rate (FPR), False Negative Rate (FNR).
- Precision-Recall Area Under Curve (PR-AUC) and Receiver Operating Characteristic (ROC-AUC).
- Inference Latency (milliseconds per flow) and Throughput (flows per second).
- Non-parametric empirical bootstrap 95% Confidence Intervals (300 replicates) and paired two-sided significance tests.

---

## 14. Results

### 14.1 Experiment A: Known Attack Detection
On test flows containing benign traffic and represented attack families:
- **XGBoost:** PR-AUC = **0.9999**, Recall = **99.99%**, Precision = **99.96%**, F1 = **0.9997**, FPR = **0.09%**.
- **VAE:** PR-AUC = **0.5974**, Recall = **0.93%**, Precision = **44.40%**, F1 = **0.0183**, FPR = **1.16%**.
- Inference latency: XGBoost achieved **0.0005 ms/flow** (1,960,000 flows/sec) versus VAE at **0.0059 ms/flow** (169,000 flows/sec).

### 14.2 Experiment B: Held-Out Attack-Family Detection (Zero-Day)
On the 132,209 unseen attack instances excluded from training:
- **DDoS (128,027 flows):** XGBoost Recall = **63.01%**; VAE Recall = **9.38%**.
- **Web Attack (2,180 flows):** XGBoost Recall = **90.23%**; VAE Recall = **0.00%**.
- **Infiltration (36 flows):** XGBoost Recall = **0.00%**; VAE Recall = **16.67%**.
- **Botnet (1,966 flows):** XGBoost Recall = **0.00%**; VAE Recall = **0.00%**.
- Overall Held-Out Recall: XGBoost = **62.50%**; VAE = **9.09%**.

### 14.3 Experiment C: False-Alarm Sensitivity Trade-Offs
Sweeping pre-declared benign false-alarm targets revealed severe anomaly score compression:
- At 0.1% target FPR: Test FPR = 0.16%, Recall = 1.57%, F1 = 0.0309.
- At 1.0% target FPR: Test FPR = 1.17%, Recall = 9.09%, F1 = 0.1660.
- At 10.0% target FPR: Test FPR = 10.25%, Recall = 9.26%, F1 = 0.1648.
Relaxing the false-alarm budget by a factor of 10 yielded only a marginal 0.17% increase in attack recall.

### 14.4 Experiment D: Distribution Shift Robustness
Under packet length scaling multipliers from $0.5\times$ to $2.0\times$:
- XGBoost PR-AUC remained exceptionally stable, degrading from 0.9991 to 0.9964.
- VAE PR-AUC exhibited steady degradation from 0.8032 down to 0.7861.

### 14.5 Experiment F: Complementarity Analysis
Among 132,209 held-out attack instances:
- Detected by Both Models: **12,005** (9.08%)
- Detected by XGBoost ONLY: **70,630** (53.42%)
- Detected by VAE ONLY: **7** (0.01% — specifically targeting Infiltration)
- Missed by Both Approaches: **49,567** (37.49%)

---

## 15. Statistical Analysis
Non-parametric paired bootstrap testing (300 iterations, 95% confidence level) confirmed:
- Known Attacks F1 Difference: $\Delta F1 = 0.9814$ (95% CI: [0.9801, 0.9827], $p < 0.001$).
- Held-Out Recall Difference: $\Delta Recall = 0.5342$ (95% CI: [0.5312, 0.5372], $p < 0.001$).
Both differences are statistically significant, confirming that XGBoost holds an empirical advantage across volumetric and structural threat classes even when unrepresented in training data.

---

## 16. Interpretability and Error Analysis
Feature attribution using XGBoost gain metrics revealed that `Destination Port`, `Init_Win_bytes_forward`, `Bwd Packet Length Max`, and `Average Packet Size` dominated decision trees. Because `DDoS` and `Web Attack` traffic exhibited similar extreme values in backward packet lengths and window allocations to trained `DoS` samples, XGBoost generalized across attack families via structural feature overlap. 

Conversely, VAE reconstruction error decomposition showed that benign traffic exhibits substantial variance in flow duration and byte rates. To maintain low benign false alarms, the VAE manifold accommodates wide flow envelopes, causing subtle low-rate attacks (such as Botnet command-and-control and Web SQL injections) to be reconstructed with low error, resulting in false negatives.

---

## 17. Discussion
The empirical results challenge the widespread cybersecurity assumption that benign anomaly detectors are inherently superior for zero-day attack detection. While the VAE functions as advertised—detecting statistical deviations without attack signatures—its real-world sensitivity is constrained by the necessity of suppressing false alarms. In high-throughput networks generating millions of flows daily, a 1% false alarm rate represents thousands of spurious alerts per hour. When calibrated to this operational constraint, the VAE misses subtle attacks. Crucially, however, the VAE demonstrated superior sensitivity to `Infiltration` attacks (16.67% vs 0.00%), establishing that anomaly detection provides vital defense-in-depth for low-footprint threats that evade supervised classifiers.

---

## 18. Limitations and Threats to Validity
1. **Dataset Synthetics:** While CICIDS2017 uses authentic traffic, its attack scenarios were executed in an emulated testbed environment that may not reflect contemporary encrypted protocol dynamics (TLS 1.3).
2. **Tabular Feature Representation:** Aggregating packet streams into bidirectional flow features discards payload and intra-flow packet arrival sequences.
3. **Hardware Execution:** Latency benchmarks reflect CPU execution; GPU batch inference would alter relative throughput dynamics.

---

## 19. Ethical Considerations
This research was conducted exclusively on publicly available, anonymized research datasets. No active network penetration, exploitation, or packet injection was performed. The models and methodologies are intended strictly for defensive security monitoring.

---

## 20. Conclusion and Future Work
This study conducted an authentic, leakage-controlled experimental comparison between supervised gradient boosting (XGBoost) and benign normal-behaviour modelling (PyTorch VAE) for network intrusion detection. Empirical evidence demonstrates that supervised classifiers can generalize across attack families when underlying flow kinematics overlap, outperforming VAEs on volumetric zero-days. However, the VAE provides unique sensitivity to stealthy infiltration threats. Future work should investigate sequence-aware temporal models (Transformers, LSTMs) and calibrated conformal anomaly scoring to control false discovery rates in live deployments.

---

## 21. References
- Arp, D., Quiring, E., Pendlebury, F., Warnecke, A., Pierazzi, F., Dos Santos, C., ... & Rieck, K. (2022). Dos and don'ts of machine learning in computer security. *Communications of the ACM*, 65(10), 74-83.
- Chen, T., & Guestrin, C. (2016). Xgboost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
- Engelen, G., Rimmer, V., & Joosen, W. (2021). Troubleshooting an incomplete machine learning flow: A cybersecurity case study. *arXiv preprint arXiv:2104.09275*.
- Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. *arXiv preprint arXiv:1312.6114*.
- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new dataset for intrusion detection: Toward developing a new dataset for intrusion detection. *ICISSP*, 108-116.

---

## 22. Appendices
- **Appendix A:** Configuration files: `configs/data.yaml`, `configs/xgboost.yaml`, `configs/vae.yaml`, `configs/experiments.yaml`.
- **Appendix B:** Audit reports: `reports/data_audit_report.json`, `reports/dataset_acquisition_summary.json`.
- **Appendix C:** Results artifact: `results/master_experiment_results.json`.
