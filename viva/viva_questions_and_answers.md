# Cognisec AI: Research Viva Defense Guide & Comprehensive Questions & Answers

**Project Title:** Cognisec AI: Network Behaviour Fingerprinting for Unseen Attack Detection — An Experimental Comparison of XGBoost and Variational Autoencoders  
**Candidate / Researcher:** Santhoskrishna  
**Repository Location:** `C:\Users\Santhoskrishna\Documents\Cognisec_AI`  
**Dataset:** Authentic CICIDS2017 Benchmark Flow Records  
**Defense Date:** October 2026  

---

## Category A: Research Problem and Novelty

### Q1: What is the core research problem addressed by Cognisec AI?
**Answer:** Traditional Network Intrusion Detection Systems (NIDS) rely predominantly on supervised discriminative algorithms (such as Random Forests, Deep Neural Networks, or Gradient Boosted Trees like XGBoost) trained on labeled datasets. While supervised classifiers achieve near-perfect detection on known attack signatures, their closed-world assumption causes catastrophic failure when encountering zero-day exploits or novel attack families whose statistical distributions diverge from the training manifold. Cognisec AI addresses this limitation by empirically benchmarking supervised tree boosting (XGBoost) against an unsupervised, one-class normal-behaviour latent density estimator (a PyTorch Tabular Variational Autoencoder, VAE) trained exclusively on benign traffic to isolate and quantify the trade-offs in unseen attack detection.

### Q2: Why use both XGBoost and a Variational Autoencoder (VAE) rather than choosing just one?
**Answer:** The two models represent fundamentally opposing learning paradigms with complementary operational properties:
1. **Supervised XGBoost (Discriminative):** Learns an optimal partitioning boundary between benign and labeled attack classes. In operational networks, it delivers ultra-low inference latency (~0.49 microseconds/flow), negligible false-alarm rates (FPR = 0.0975%), and 99.99% recall on familiar attack vectors.
2. **Tabular PyTorch VAE (Generative/Density-based):** Learns the underlying continuous probability density distribution of normal benign traffic alone. Because it is never trained on attack samples, it is theoretically immune to attack signature overfitting and capable of flagging structural behavioral deviations.
Benchmarking both paradigms under identical data splits reveals their true operational trade-offs, proving that neither model is universally superior on its own.

### Q3: What is the primary novelty of this research compared to standard IDS benchmark papers?
**Answer:** Most published IDS literature trains and tests models on random stratified splits where all attack families appear in both training and test partitions, creating an illusion of high generalization. Cognisec AI introduces a rigorous, leakage-controlled **Held-Out Attack Family Protocol** where four distinct threat categories (`DDoS`, `Web_Attack`, `Bot`, `Infiltration`) are completely excluded from model training and hyperparameter tuning. Furthermore, we provide a mathematically reconciled audit distinguishing physical dataset volume (257,297 unique records) from total benchmark evaluations (297,297 evaluations across two controlled experiments), accompanied by paired bootstrap and McNemar statistical tests.

---

## Category B: Dataset and Preprocessing

### Q4: What is the provenance and official source of the CICIDS2017 dataset?
**Answer:** CICIDS2017 was created and published by the Canadian Institute for Cybersecurity (CIC) at the University of New Brunswick (UNB) by Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani (ICISSP 2018). It captures five consecutive days of network traffic (Monday July 3, 2017 to Friday July 7, 2017) generated in a controlled testbed environment consisting of attacking machines (Kali Linux) and targeted victim subnetworks. The traffic was recorded in raw PCAP format and converted into bidirectional flow records using CICFlowMeter.

### Q5: What raw CSV files are present in the project, and how were they cleaned?
**Answer:** The repository contains all 8 authentic daily CICFlowMeter CSV files:
1. `Monday-WorkingHours.pcap_ISCX.csv` (Baseline benign traffic)
2. `Tuesday-WorkingHours.pcap_ISCX.csv` (FTP/SSH Patator brute force)
3. `Wednesday-workingHours.pcap_ISCX.csv` (DoS slowloris, SlowHTTPTest, Hulk, GoldenEye, Heartbleed)
4. `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` (Brute Force, XSS, SQL Injection)
5. `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` (MetaSploit Infiltration)
6. `Friday-WorkingHours-Morning.pcap_ISCX.csv` (Botnet Ares)
7. `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` (Firewall port scanning)
8. `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (LOIC DDoS attack)

**Cleaning steps:**
- Column name whitespace trimming and lowercasing.
- Removal of 8 non-behavioral identity columns: `Flow ID`, `Source IP`, `Source Port`, `Destination IP`, `Destination Port`, `Protocol`, `Timestamp`, and `External IP`.
- Filtering non-finite numeric values: infinite values generated by division-by-zero in flow duration were replaced with median upper bounds; missing/NaN values (~0.01%) were imputed using training-set feature medians.
- Label normalization into binary flags (`0` for Benign, `1` for Attack) alongside multiclass family preservation.

### Q6: How many features were retained in the preprocessor, and how was feature selection conducted?
**Answer:** Exactly **70 behavioral statistical flow features** were retained in `artifacts/models/flow_preprocessor.joblib`. Feature selection was conducted strictly on training data by discarding socket identifiers, timestamps, zero-variance columns (e.g., `Bwd PSH Flags`, `Fwd URG Flags`), and duplicate flow duration metrics. The retained features capture temporal intervals (inter-arrival times), packet size distributions, flow volume dynamics, TCP header flag counts, and sub-flow rates.

### Q7: How was data leakage prevented during the preprocessing pipeline?
**Answer:** Preprocessing transformations (median imputation and `RobustScaler` / `StandardScaler` quantile statistics) were fitted **strictly on the training split** and persisted into `artifacts/models/flow_preprocessor.joblib`. Test and validation partitions were transformed using the fitted preprocessor without updating parameters. Furthermore, identity features (IP addresses, ports, and timestamps) were purged prior to feature scaling, ensuring models could not memorize host topologies or temporal attack schedules.

---

## Category C: XGBoost Theory and Configuration

### Q8: What exact hyperparameter configuration was used for the supervised XGBoost model?
**Answer:** Inspected from `artifacts/models/xgboost_model.json` and training logs:
- **Algorithm:** Extreme Gradient Boosting (`XGBClassifier`)
- **Objective:** `binary:logistic`
- **Number of Estimators (`n_estimators`):** 200 trees
- **Maximum Tree Depth (`max_depth`):** 6
- **Tree Method (`tree_method`):** Histogram-based (`hist`), enabling rapid quantile binning
- **Learning Rate (`learning_rate` / `eta`):** 0.10
- **Subsample Ratio:** 0.80 (row subsampling to prevent overfitting)
- **Colsample By Tree:** 0.80 (feature subsampling per tree)
- **Early Stopping:** 15 validation rounds on log-loss
- **Calibrated Decision Threshold:** `0.52` (selected via validation set grid search maximizing $F_1$-score)

### Q9: How does XGBoost compute flow probabilities and make classifications?
**Answer:** Each decision tree outputs a continuous leaf value (raw log-odds score). The ensemble aggregates the raw outputs across all 200 boosted trees:
$$\hat{y}_{\text{raw}} = \sum_{k=1}^K f_k(\mathbf{x})$$
The aggregated log-odds are mapped to a well-calibrated class probability via the logistic sigmoid function:
$$P(\text{Attack} \mid \mathbf{x}) = \sigma(\hat{y}_{\text{raw}}) = \frac{1}{1 + e^{-\hat{y}_{\text{raw}}}}$$
If $P(\text{Attack} \mid \mathbf{x}) \ge 0.52$, the flow is flagged as an Attack; otherwise, it is classified as Benign.

---

## Category D: VAE Theory, Latent Space, and Loss Formulation

### Q10: What is the architecture of the Tabular PyTorch Variational Autoencoder?
**Answer:** Inspected from `artifacts/models/vae_checkpoint.pt`:
- **Input Dimension:** 70 continuous behavioral features.
- **Encoder:** Dense feedforward network: $\text{Linear}(70 \to 64) \to \text{BatchNorm1d} \to \text{LeakyReLU}(0.2) \to \text{Linear}(64 \to 32) \to \text{BatchNorm1d} \to \text{LeakyReLU}(0.2)$.
- **Latent Bottleneck:** Two parallel linear heads:
  - Mean vector: $\boldsymbol{\mu} \in \mathbb{R}^{16}$
  - Log-variance vector: $\log \boldsymbol{\sigma}^2 \in \mathbb{R}^{16}$
- **Reparameterization Trick:** $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$, where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{16})$.
- **Decoder:** Symmetric reconstruction network: $\text{Linear}(16 \to 32) \to \text{BatchNorm1d} \to \text{LeakyReLU}(0.2) \to \text{Linear}(32 \to 64) \to \text{BatchNorm1d} \to \text{LeakyReLU}(0.2) \to \text{Linear}(64 \to 70)$.

### Q11: What is the exact loss function used to train the VAE?
**Answer:** The VAE is trained to minimize the $\beta$-Variational Evidence Lower Bound ($\beta$-ELBO) loss:
$$\mathcal{L}_{\text{VAE}}(\mathbf{x}, \hat{\mathbf{x}}) = \mathcal{L}_{\text{Recon}}(\mathbf{x}, \hat{\mathbf{x}}) + \beta \cdot D_{\text{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x}) \parallel p(\mathbf{z})\big)$$
Where:
1. **Reconstruction Loss:** Mean Squared Error across all 70 normalized features:
   $$\mathcal{L}_{\text{Recon}}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{70} \sum_{j=1}^{70} (x_j - \hat{x}_j)^2$$
2. **Kullback-Leibler Divergence:** Analytic divergence between the encoder posterior and the standard Gaussian prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$:
   $$D_{\text{KL}} = -\frac{1}{2} \sum_{i=1}^{16} \left(1 + \log \sigma_i^2 - \mu_i^2 - \sigma_i^2\right)$$
3. **KL Weighting ($\beta$):** Set to $\beta = 0.005$ to prevent posterior collapse on high-dimensional tabular data.

### Q12: Why was the VAE trained exclusively on benign traffic?
**Answer:** Training the VAE exclusively on benign traffic enforces the **one-class anomaly detection formulation**. The model learns the compact manifold and feature correlations characterizing legitimate network behavior. When an anomalous attack flow traverses the autoencoder, its feature combinations violate the learned correlations, causing high reconstruction error (MSE). Crucially, this enables the VAE to potentially detect zero-day attacks without requiring prior exposure to attack signatures during training.

---

## Category E: Anomaly Scores and False-Positive Trade-offs

### Q13: What is the difference between a decision threshold and an anomaly score?
**Answer:** 
- An **anomaly score** is a continuous scalar metric measuring how severely an observed network flow deviates from expected normal behavior. In our Tabular VAE, the anomaly score is the per-flow Mean Squared Reconstruction Error: $s(\mathbf{x}) = \frac{1}{D} \|\mathbf{x} - g_\theta(f_\phi(\mathbf{x}))\|_2^2$.
- A **decision threshold** ($\tau$) is an operational decision boundary that maps continuous scores into a binary classification decision: $\hat{y} = \mathbb{I}(s(\mathbf{x}) \ge \tau)$.

### Q14: How was the VAE decision threshold of 0.330610 selected?
**Answer:** In network intrusion detection, false alarm fatigue is critical; security operations centers cannot tolerate false alarm rates exceeding 1%. The VAE threshold was calibrated on the **benign validation split** by identifying the score value at the 99th percentile:
$$\tau = \text{Percentile}\left(\{s(\mathbf{x}_{\text{val, benign}})\}, 99.0\right) = 0.330610$$
This guarantees an operational False Positive Rate ceiling of at most 1.0% on uncorrupted baseline traffic.

### Q15: Why might a VAE exhibit high precision but low recall on network data?
**Answer:** In Experiment B (Held-Out Attacks), the VAE achieved 98.33% precision but only 16.60% recall. This occurs because the calibrated threshold ($\tau = 0.330610$) was tuned conservatively to limit false alarms to under 1%. While this ensures that flows exceeding the threshold are overwhelmingly true attacks (high precision), a significant portion of unseen attack flows produce reconstruction errors overlapping with the heavy tail of benign traffic ($s < 0.330610$), causing high false-negative rates (low recall).

---

## Category F: Experimental Design and Unseen Attack Families

### Q16: What does “unseen attack” mean in this experimental design?
**Answer:** In our experimental design, "unseen attack" refers to a **Held-Out Threat Family** (`DDoS`, `Web_Attack`, `Bot`, `Infiltration`) whose samples were completely excluded from the training split ($X_{\text{train}}, y_{\text{train}}$) and validation split ($X_{\text{val}}, y_{\text{val}}$). The models had zero exposure to these specific attack classes during weight optimization and threshold tuning.

### Q17: Does detecting a held-out family prove generalization to truly novel real-world zero-day attacks?
**Answer:** **No, it does not prove universal zero-day generalization, and claiming so would be an academic overstatement.** While holding out families simulates structural novelty better than random cross-validation, the test samples still originate from the same testbed environment, IP subnet topology, and operating systems as the training data. Furthermore, some held-out families share behavioral mechanisms with training classes (e.g., `DDoS` uses volumetric TCP/UDP flooding identical to `DoS Hulk` in the training set). True zero-day generalization can only be validated against entirely independent external datasets (e.g., CIC-CSE-IDS2018 or operational network traffic).

### Q18: What were the exact sample counts in Experiment A and Experiment B?
**Answer:**
- **Experiment A (Known Attacks Benchmark):** 125,088 flows (40,000 Benign + 85,088 Known Attacks from DoS, PortScan, and Brute Force).
- **Experiment B (Held-Out Unseen Benchmark):** 172,209 flows (40,000 Benign control + 132,209 Held-Out Attacks from DDoS, Web Attack, Bot, and Infiltration).
- **Shared Benign Baseline:** The exact same 40,000 benign control flows were evaluated in both experiments to provide a constant false-alarm baseline.
- **Physical Test Set Volume:** $40,000 + 85,088 + 132,209 = 257,297$ unique flow records. Total benchmark evaluation decisions equal $125,088 + 172,209 = 297,297$.

---

## Category G: Metrics, Confusion Matrices, and Class Imbalance

### Q19: Why is overall classification accuracy a misleading metric in intrusion detection?
**Answer:** Classification accuracy calculates $\frac{\text{TP} + \text{TN}}{\text{Total}}$. In highly imbalanced operational network traffic where benign flows constitute 99% of all traffic, a trivial "dummy" classifier that predicts all flows as benign achieves 99% accuracy while having a 0% detection rate for attacks. Accuracy obscures critical false negatives.

### Q20: Why are Precision-Recall curves (PR-AUC) superior to ROC curves (ROC-AUC) for IDS evaluation?
**Answer:** ROC-AUC plots True Positive Rate vs. False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$). When the benign class ($\text{TN}$) is extremely large, the denominator in FPR is huge, causing FPR to remain deceptively small even when thousands of false alarms occur. In contrast, Precision ($\frac{\text{TP}}{\text{TP} + \text{FP}}$) directly penalizes every false positive relative to true detections. Thus, PR-AUC provides a much harsher, more operationally realistic measure of performance under severe class imbalance.

### Q21: What are the exact confusion matrix numbers for XGBoost and VAE in Experiment A?
**Answer:**
- **XGBoost:** $\text{TN} = 39,961$, $\text{FP} = 39$, $\text{FN} = 8$, $\text{TP} = 85,080$ (Total = 125,088; Recall = 99.99%, Precision = 99.95%).
- **VAE:** $\text{TN} = 39,628$, $\text{FP} = 372$, $\text{FN} = 63,058$, $\text{TP} = 22,030$ (Total = 125,088; Recall = 25.89%, Precision = 98.34%).

### Q22: What are the exact confusion matrix numbers in Experiment B?
**Answer:**
- **XGBoost:** $\text{TN} = 39,961$, $\text{FP} = 39$, $\text{FN} = 49,574$, $\text{TP} = 82,635$ (Total = 172,209; Recall = 62.50%, Precision = 99.95%).
- **VAE:** $\text{TN} = 39,628$, $\text{FP} = 372$, $\text{FN} = 110,268$, $\text{TP} = 21,941$ (Total = 172,209; Recall = 16.60%, Precision = 98.33%).

---

## Category H: Statistical Significance and Paired Testing

### Q23: Why is reporting an approximate p-value of 0.0 statistically suspicious?
**Answer:** In inferential statistics, a true p-value is a continuous probability bounded in $(0, 1]$. Software packages often print `0.0` or `0.0000` because the calculated value falls below floating-point machine precision (e.g., $p = 10^{-16}$). Reporting $p = 0.0$ in academic literature is mathematically improper as it suggests an impossible zero probability. The correct scientific notation is to report a justified upper bound, such as $p < 0.001$.

### Q24: What statistical test is appropriate for comparing paired model classifications?
**Answer:** Because both XGBoost and the VAE evaluate the exact same test instances ($N = 125,088$ in Exp A and $N = 172,209$ in Exp B), their prediction outcomes are strictly **paired binary observations**. McNemar’s test (a 1-degree-of-freedom $\chi^2$ test on the discordant cell matrix) is the standard nonparametric test:
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
Where $b$ is the number of flows correctly classified by Model 1 but missed by Model 2, and $c$ is the reverse. For performance metric differences ($\Delta F_1$ and $\Delta \text{Recall}$), paired non-parametric bootstrap resampling ($B = 1,000$ iterations) provides bias-corrected 95% confidence intervals.

### Q25: What were the results of the McNemar test and paired bootstrap?
**Answer:**
- **Experiment A:**
  - McNemar's $\chi^2 = 63,305.09$ ($p < 0.001$), confirming a statistically significant difference in classification decisions.
  - Paired Bootstrap $\Delta F_1 = +0.5898$ (95% CI: `[0.5859, 0.5934]`, $p < 0.001$).
- **Experiment B:**
  - McNemar's $\chi^2 = 60,128.40$ ($p < 0.001$).
  - Paired Bootstrap $\Delta \text{Recall} = +0.4591$ (95% CI: `[0.4562, 0.4619]`, $p < 0.001$).

---

## Category I: Data Leakage, Duplicate Flows, and Validity Threats

### Q26: How do duplicate flows cause data leakage, and how was this audited?
**Answer:** Network flow datasets frequently contain duplicate or near-identical flow vectors generated by repeated background daemon polling or repetitive ping sweeps. If identical vectors appear in both the training set and the test set, supervised classifiers can achieve inflated test scores by memorizing training instances. Our audit verified that:
1. 14,682 identical vectors existed between raw training and test pools.
2. We verified that held-out attack families had **zero representation** in the training partition.
3. IP addresses, ports, and timestamps were eliminated from the feature schema so socket-level memorization was impossible.

### Q27: What is the primary threat to validity regarding XGBoost's 62.50% recall on held-out attacks?
**Answer:** The primary threat is **volumetric signature leakage**. In Experiment B, 128,027 of the 132,209 held-out attacks (96.8%) belong to the `DDoS` family. XGBoost achieved 63.01% recall on `DDoS` because it was trained on `DoS` attacks (which exhibit near-identical high packet rates and short inter-arrival times). When evaluated on the truly stealthy `Infiltration` family (36 flows), XGBoost suffered **0.0% recall**, exposing its complete vulnerability when attack mechanics truly diverge from training data.

---

## Category J: Flask Architecture, Model Artifacts, and CSV Inference

### Q28: How does the application architecture process an uploaded CSV file?
**Answer:**
1. **Frontend:** Vanilla JavaScript captures the user's file or loads the sample flow set and posts it via `fetch('/api/analyze/xgboost')` or `/api/analyze/vae`.
2. **Backend Validation:** Flask receives the multipart file stream, checks CSV headers against the 70 required features via `FlowPreprocessor.validate_dataframe()`, and handles missing/infinite numbers.
3. **Preprocessing:** The DataFrame is scaled using the fitted `flow_preprocessor.joblib`.
4. **Inference:**
   - XGBoost predicts class probabilities via `xgb_model.predict_proba()`.
   - VAE computes latent representations and Mean Squared Reconstruction Error on PyTorch tensors.
5. **Response:** Flask returns structured JSON containing per-flow scores, calibrated threshold comparisons, and summary metrics.

### Q29: Why must the system be described as an offline CSV analyzer rather than a live network monitor?
**Answer:** The backend processes pre-computed flow summary CSVs exported from flow extractors like CICFlowMeter. It does not bind to raw network interfaces (e.g., via `libpcap` or `scapy`), does not assemble TCP packets into flows in real time, and does not perform streaming sliding-window analysis. Accurately describing it as an offline research benchmark preserves academic and technical integrity.

---

## Category K: Results Interpretation and Complementarity

### Q30: What does the complementarity analysis reveal between XGBoost and the VAE?
**Answer:** Among the 132,209 held-out attack flows in Experiment B:
- **Both detected:** 21,521 flows (16.28%)
- **XGBoost only:** 61,114 flows (46.23%)
- **VAE only:** 420 flows (0.32%)
- **Both missed:** 49,154 flows (37.18%)

While XGBoost detected more total flows due to high-volume DDoS, the VAE uniquely detected 47.22% of stealthy `Infiltration` attacks where XGBoost achieved 0.00% recall. This demonstrates that a normal-behaviour VAE provides orthogonal detection coverage on non-volumetric attacks that evade supervised boundaries.

### Q31: How did distribution shift affect the two models in Experiment D?
**Answer:** Controlled feature scaling factors ($0.5\times$ to $2.0\times$) were applied to test features.
- XGBoost PR-AUC shifted from 0.9991 down to 0.9964 (a degradation of 0.27%).
- VAE PR-AUC shifted from 0.9842 down to 0.9810 (a degradation of 0.32%).
Both models demonstrated stability under proportional scaling, but this controlled simulation does not replace validation on genuine out-of-distribution traffic from modern networks.

---

## Category L: Limitations, Operational Deployment, and Future Work

### Q32: What are the main limitations of the CICIDS2017 dataset?
**Answer:**
1. **Age:** Captured in 2017, it lacks modern threat vectors such as TLS 1.3 encrypted malware, DNS-over-HTTPS exfiltration, and contemporary ransomware.
2. **Synthetic Traffic Generation:** Background benign traffic was simulated using B-Profile tools rather than genuine human enterprise workflows.
3. **CICFlowMeter Artifacts:** Known bugs in older CICFlowMeter releases generate corrupted packet length statistics and infinite duration values.

### Q33: What would be required to deploy Cognisec AI into a production enterprise network?
**Answer:**
1. A streaming packet-capture engine (e.g., DPDK, Zeek, or eBPF) capable of generating flow statistics in microsecond windows.
2. Continual retraining pipelines with automated drift detection to update the VAE's benign manifold as enterprise software updates occur.
3. An ensemble or tiered decision-fusion policy (e.g., routing flows through XGBoost first for sub-microsecond filtering, and escalating borderline flows to the VAE).
4. Strict operational latency budgets and hardened inference APIs.

### Q34: Why was Heartbleed detected with 100% recall by both models, and why must we be cautious interpreting it?
**Answer:** The test partition contains only **3 records** of Heartbleed. While both models achieved 100% recall, drawing broad statistical conclusions from an $N=3$ sample size is scientifically invalid due to extreme sample variance.

### Q35: What is the final conclusion of this research project?
**Answer:** Supervised classifiers like XGBoost remain superior in operational settings for high-volume, signature-represented attacks due to sub-microsecond latency and near-zero false alarms. However, they possess a fatal blind spot for subtle, structurally novel intrusions (0.0% on Infiltration). Unsupervised VAEs trained on benign traffic provide crucial orthogonal visibility into subtle anomalies, but operate under severe false-alarm trade-offs. The optimal architecture for next-generation NIDS is a hybrid, multi-tier deployment leveraging supervised tree boosting for frontline throughput alongside generative latent modeling for zero-day anomaly triage.
