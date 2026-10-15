# Cognisec AI — Comprehensive Experimental Results and Benchmarks

**Project:** Cognisec AI: Network Behaviour Fingerprinting for Unseen Attack Detection — An Experimental Comparison of XGBoost and Variational Autoencoders  
**Evaluation Dataset:** Authentic CICIDS2017 Test Partitions (`data/processed/test_data.npz`)  
**Results Source:** `results/master_experiment_results.json` and `results/reproducibility_audit_report.json`  

---

## 1. Primary Benchmark Findings

### 1.1 Experiment A: Known Attack Family Benchmark (125,088 Test Flows)
*Test split containing Benign traffic (40,000) and attack families represented during training (`DoS`, `PortScan`, `Brute_Force`, `Heartbleed`: 85,088).*

| Evaluation Metric | XGBoost (Supervised) | Tabular VAE (Unsupervised) | Paired Difference ($\Delta$) | Statistical Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Precision** | **99.9542%** | **98.3394%** | +1.6148% | Statistically significant ($p < 0.001$) |
| **Recall (Detection Rate)** | **99.9906%** | **25.8908%** | +74.0998% | Statistically significant ($p < 0.001$) |
| **F1-Score** | **0.9997** | **0.4099** | **+0.5898** | 95% CI: `[0.5859, 0.5934]`, $p < 0.001$ |
| **PR-AUC** | **0.999979** | **0.943188** | +0.056791 | High discrimination across thresholds |
| **ROC-AUC** | **0.999959** | **0.914521** | +0.085438 | Broad separability |
| **False Positive Rate (FPR)** | **0.0975%** (39 FP) | **0.9300%** (372 FP) | -0.8325% | VAE strictly satisfies 1.0% budget |
| **True Negatives (TN)** | 39,961 / 40,000 | 39,628 / 40,000 | +333 | Benign baseline concordance |
| **True Positives (TP)** | 85,080 / 85,088 | 22,030 / 85,088 | +63,050 | High tree sensitivity to known patterns |
| **False Negatives (FN)** | 8 / 85,088 | 63,058 / 85,088 | -63,050 | Low-variance benign envelope causes miss |
| **CPU Latency (per flow)** | **0.00048 ms** | **0.00399 ms** | 8.3x faster | 2.08M flows/sec vs 250.9K flows/sec |

---

### 1.2 Experiment B: Held-Out Attack Family Benchmark (172,209 Test Flows)
*Test split containing Benign traffic (40,000) and completely unseen attack families excluded from training (`DDoS`, `Web_Attack`, `Bot`, `Infiltration`: 132,209).*

| Evaluation Metric | XGBoost (Supervised) | Tabular VAE (Unsupervised) | Paired Difference ($\Delta$) | Statistical Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Precision** | **99.9528%** | **98.3328%** | +1.6200% | High precision maintained by both |
| **Recall (Detection Rate)** | **62.5033%** | **16.5957%** | **+0.4591** | 95% CI: `[0.4562, 0.4619]`, $p < 0.001$ |
| **F1-Score** | **0.7691** | **0.2840** | +0.4851 | Substantial drop for both models |
| **PR-AUC** | **0.992398** | **0.980167** | +0.012231 | High potential across sweep |
| **ROC-AUC** | **0.979248** | **0.962920** | +0.016328 | High continuous separation |
| **False Positive Rate (FPR)** | **0.0975%** (39 FP) | **0.9300%** (372 FP) | -0.8325% | Identical benign behavior |
| **True Negatives (TN)** | 39,961 / 40,000 | 39,628 / 40,000 | +333 | Consistent false-alarm control |
| **True Positives (TP)** | 82,635 / 132,209 | 21,941 / 132,209 | +60,694 | Volumetric attacks dominate count |
| **False Negatives (FN)** | 49,574 / 132,209 | 110,268 / 132,209 | -60,694 | Stealthy zero-days evade single models |

---

## 2. Disaggregated Attack-Family Recall Analysis (257,297 Test Flows)

Evaluating all attack classes across the entire test set reveals the central qualitative insight of this research:

| Attack Family | Status | Test Flow Count | XGBoost Recall | VAE Recall | Key Kinematic & Detection Finding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Infiltration** | **Held-Out (Unseen)** | 36 | **0.00%** | **47.22%** | **VAE uniquely detects subtle lateral movement and port manipulation where XGBoost has 0.0% recall.** |
| **Bot** | **Held-Out (Unseen)** | 1,966 | **0.00%** | **0.56%** | Periodic low-rate C2 heartbeats blend into normal HTTP traffic; evades both systems. |
| **DDoS** | **Held-Out (Unseen)** | 128,027 | **63.01%** | **17.12%** | High backward packet length and burst rates structurally transfer from trained `DoS` to `DDoS`. |
| **Web Attack** | **Held-Out (Unseen)** | 2,180 | **90.23%** | **0.00%** | SQLi/Brute Force payload attempts exhibit distinctive TCP window and header features flagged by trees. |
| **DoS** | **Known (Represented)** | 50,645 | **99.99%** | **43.43%** | Large packet size variance triggers high VAE reconstruction error; near-perfect tree recognition. |
| **PortScan** | **Known (Represented)** | 31,666 | **99.99%** | **0.10%** | Single SYN probes generate low reconstruction error, causing high VAE false negatives at 1% FPR. |
| **Brute Force** | **Known (Represented)** | 2,774 | **99.96%** | **0.00%** | Repetitive authentication flows mimic benign connection negotiation. |
| **Heartbleed** | **Known (Represented)** | 3 | **100.00%** | **100.00%** | Pathological packet length anomaly (large buffer over-read) flagged by both models. |

---

## 3. Threshold Sensitivity Analysis (Experiment C)

Evaluating the VAE anomaly score threshold $\tau$ across pre-declared benign false-alarm targets on the test partition:

| Target FPR | Threshold ($\tau$) | Empirical Test FPR | Attack Recall | Precision | F1-Score | Operational Trade-off Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.1%** | `0.785674` | **0.13%** | 0.01% | 18.75% | 0.0002 | Ultra-conservative; misses almost all attacks |
| **0.5%** | `0.427767` | **0.48%** | 9.13% | 98.43% | 0.1671 | Low alert volume, moderate volumetric recall |
| **1.0% (Default)** | `0.330610` | **0.93%** | 16.60% | 98.33% | 0.2840 | **Operational balanced baseline; 1% false alarm ceiling** |
| **2.0%** | `0.264008` | **1.89%** | 42.29% | 98.66% | 0.5920 | Doubles recall with acceptable alert increase |
| **5.0%** | `0.172278` | **5.03%** | **86.17%** | **98.26%** | **0.9182** | **Massive recall surge (+69.57%) at cost of 5% false alarms** |
| **10.0%** | `0.117411` | **10.14%** | **93.66%** | 96.83% | 0.9522 | High sensitivity, but unviable for high-speed SOC |

---

## 4. Model Complementarity Analysis (Experiment F)

Cross-tabulating predictions on the 132,209 unseen held-out attack instances (`DDoS`, `Web_Attack`, `Bot`, `Infiltration`):

```
                        Held-Out Attack Decisions (132,209 Flows)
┌──────────────────────────────────────┬──────────────────────────────────────┐
│ Both Models Detected: 21,521 (16.28%) │ XGBoost Only Detected: 61,114 (46.23%)│
├──────────────────────────────────────┼──────────────────────────────────────┤
│ VAE Only Detected:       420 (0.32%) │ Both Models Missed:    49,154 (37.18%)│
└──────────────────────────────────────┴──────────────────────────────────────┘
```
- **Mutual Exclusivity:** $21,521 + 61,114 + 420 + 49,154 = 132,209$ (100.00% verified).
- **Ensemble Potential:** If an operational hybrid system triggers an alert when *either* model flags an anomaly ($\text{OR}$-gate logic), the combined held-out recall rises from **62.50% to 62.82%**, capturing 420 additional stealth attacks (including 17 Infiltration instances) without requiring attack signatures.

---

## 5. Distribution-Shift Sensitivity (Experiment E / D)

Perturbing packet length metrics by multiplicative scaling factors $k \in \{0.5, 0.8, 1.0, 1.2, 1.5, 2.0\}$:

| Perturbation Multiplier ($k$) | XGBoost PR-AUC | Tabular VAE PR-AUC | Resilience Assessment |
| :--- | :--- | :--- | :--- |
| **0.5x (Compressed Packets)** | 0.9991 | 0.9842 | Both robust; trees slightly unaffected |
| **0.8x** | 0.9996 | 0.9843 | Minimal variance in latent reconstruction |
| **1.0x (Baseline)** | 0.9975 | 0.9839 | Reference unshifted performance |
| **1.2x** | 0.9973 | 0.9833 | Smooth degradation |
| **1.5x** | 0.9974 | 0.9824 | Continuous manifold maintains stability |
| **2.0x (Expanded Packets)** | 0.9964 | 0.9810 | High bounded resilience ($>0.98$ PR-AUC) |
