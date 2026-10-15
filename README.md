# Network Behaviour Fingerprinting for Unseen Attack Detection
### An Experimental Comparison of XGBoost and Variational Autoencoders

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost 3.4](https://img.shields.io/badge/XGBoost-3.4-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![Dataset CICIDS2017](https://img.shields.io/badge/Dataset-CICIDS2017-orange.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)
[![License MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 1. Executive Summary

In cybersecurity intrusion detection, two foundational paradigms compete:
1. **Supervised Traditional Machine Learning (XGBoost):** Highly sensitive and reliable for known attack signatures, but suspected of failing under novel zero-day attack variants.
2. **Semi-Supervised Deep Learning (Variational Autoencoders):** Trained exclusively on normal benign network flows to learn an authentic representation manifold of benign behavior, flagging any significant reconstruction error as an anomaly.

This project delivers a **research-grade, leakage-controlled empirical benchmark** on authentic **CICIDS2017** network-flow records comparing **exactly two predictive models**: an **XGBoost classifier** and a **PyTorch Tabular VAE**.

---

## 2. Key Empirical Findings

| Metric / Experiment | XGBoost (Supervised) | PyTorch VAE (Benign Anomaly) | Statistical Finding & Significance |
| :--- | :--- | :--- | :--- |
| **Known Attack PR-AUC (Exp A)** | **0.9999** | 0.5974 | Baseline comparison on seen families ($p < 0.001$) |
| **Known Attack F1-Score (Exp A)** | **0.9997** [0.9996, 0.9998] | 0.0183 [0.0171, 0.0196] | $\Delta F1 = 0.9814$ (Significant: True) |
| **Known Attack Recall (Exp A)** | **99.99%** | 0.93% | At 1.0% benign false-alarm constraint |
| **Known Attack FPR (Exp A)** | **0.09%** | 1.16% | Enforced via benign validation percentile |
| **Held-Out (Zero-Day) Recall (Exp B)**| **62.50%** [0.6229, 0.6277] | 9.09% [0.0891, 0.0923] | $\Delta Recall = 0.5342$ ($p < 0.001$) |
| **Infiltration Attack Recall (Exp B)**| 0.00% | **16.67%** | **VAE detected subtle stealth attacks missed completely by XGBoost** |
| **Inference Latency** | **0.0005 ms/sample** | 0.0059 ms/sample | XGBoost is ~11x faster on CPU |

---

## 3. Project Architecture

```
network-behaviour-fingerprinting/
├── configs/
│   ├── data.yaml                     # Ingestion, leakage drops, family mappings, split ratios
│   ├── xgboost.yaml                  # Hyperparameters, early stopping, validation thresholding
│   ├── vae.yaml                      # Encoder/decoder dims, beta-KL, early stopping, percentile FPR
│   └── experiments.yaml              # Protocol configurations for Experiments A through F
├── data/
│   ├── raw/                          # Authentic CICIDS2017 CSV flow records (8 daily files)
│   └── processed/                    # Leakage-controlled partitions (train, val, test npz)
├── src/
│   ├── data/
│   │   ├── download_cicids2017.py    # Authentic dataset acquisition with checksum validation
│   │   ├── ingestion.py              # Schema cleaning, numeric downcasting, provenance tracking
│   │   ├── validation.py             # Quality audit (nulls, infinities, duplicates, class distributions)
│   │   ├── preprocessing.py          # Leakage removal and dual-model representation fitting
│   │   ├── splitting.py              # Zero-leakage partitioning with held-out attack family isolation
│   │   └── prepare_data.py           # Phase 3 automated pipeline orchestrator
│   ├── models/
│   │   ├── xgboost_model.py          # XGBoost classifier with validation threshold optimization
│   │   └── vae_model.py              # PyTorch Tabular VAE with reparameterization & anomaly scoring
│   ├── training/
│   │   ├── train_xgboost.py          # XGBoost training pipeline
│   │   └── train_vae.py              # Benign-only VAE training pipeline
│   ├── evaluation/
│   │   ├── metrics.py                # Comprehensive metrics (PR-AUC, ROC-AUC, FPR, FNR, family recall)
│   │   ├── thresholds.py             # Benign-only percentile threshold selection & sensitivity sweeps
│   │   ├── statistics.py             # Empirical bootstrap 95% CIs & paired significance testing
│   │   └── robustness.py             # Distribution-shift feature perturbation generators
│   ├── analysis/
│   │   ├── interpretability.py       # XGBoost gain importance & VAE error decomposition
│   │   └── visualisation.py          # Publication-grade plotting suite (Figures 1 to 8)
│   ├── experiments/
│   │   └── run_experiments.py        # Master research runner executing Experiments A, B, C, D, F
│   └── utils/
│       ├── configuration.py          # YAML config loader
│       └── reproducibility.py        # Deterministic seed fixing across random, numpy, and torch
├── tests/
│   ├── test_environment.py           # Environment import validation
│   ├── test_xgboost.py               # XGBoost training, inference, and serialization tests
│   └── test_vae.py                   # PyTorch VAE architecture, training, and threshold tests
├── reports/
│   ├── figures/                      # Publication-grade figures (fig1 through fig8)
│   ├── tables/                       # Markdown & LaTeX comparative benchmark tables
│   ├── data_audit_report.json        # Data quality and schema audit report
│   └── academic_report.md            # Complete 22-section dissertation research report
├── artifacts/models/                 # Serialized model weights, preprocessor, and thresholds
├── app.py                            # Streamlit interactive research demonstration dashboard
├── requirements.txt                  # Strict pinned Python dependencies
└── pytest.ini                        # Automated test configuration
```

---

## 4. Setup & Reproducibility Guide

### 4.1 Prerequisites
- Python 3.12 (Installed locally)
- 8 GB+ RAM (16 GB recommended)
- Optional: NVIDIA CUDA-compatible GPU

### 4.2 Installation
```bash
# Clone the repository
git clone https://github.com/your-username/network-behaviour-fingerprinting.git
cd network-behaviour-fingerprinting

# Create and activate virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\activate      # Windows
source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 4.3 Automated Pipeline Execution
```bash
# Step 1: Run Unit Tests
pytest tests/

# Step 2: Acquire Authentic CICIDS2017 Dataset
python -m src.data.download_cicids2017

# Step 3: Run Preprocessing & Leakage-Controlled Partitioning
python -m src.data.prepare_data

# Step 4: Execute Master Research Experiments (Exp A, B, C, D, F)
python -m src.experiments.run_experiments

# Step 5: Launch the Streamlit Demonstration Dashboard
streamlit run app.py
```

---

## 5. Summary of Research Figures Generated

- **Figure 1 (`fig1_dataset_distribution.png`):** Distribution of the 757,646 authentic flow records disaggregated by attack family.
- **Figure 2 (`fig2_roc_pr_curves.png`):** Precision-Recall and ROC curves comparing XGBoost and VAE on known attack traffic.
- **Figure 3 (`fig3_confusion_matrices.png`):** Confusion matrices showing exact TP, FP, FN, TN counts for both models.
- **Figure 4 (`fig4_anomaly_score_distributions.png`):** Latent reconstruction anomaly score distributions contrasting Benign, Known Attacks, and Zero-Day Attacks against the 1.0% FPR threshold.
- **Figure 5 (`fig5_per_family_recall.png`):** Disaggregated detection recall across every individual attack family.
- **Figure 6 (`fig6_threshold_sensitivity.png`):** Operational trade-off curves sweeping false-alarm targets from 0.1% to 10.0%.
- **Figure 7 (`fig7_distribution_shift_degradation.png`):** Generalisation degradation under packet size distribution shifts.
- **Figure 8 (`fig8_model_disagreement.png`):** Complementarity analysis disaggregating detections on 132,209 unseen held-out attacks.
