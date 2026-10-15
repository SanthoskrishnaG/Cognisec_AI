# Cognisec AI: Network Behaviour Fingerprinting for Unseen Attack Detection
### An Experimental Comparison of XGBoost and Variational Autoencoders
**Tagline:** *Intelligent Network Behaviour Analysis for Unseen Attack Detection*

<p align="center">
  <img src="frontend/assets/logo.svg" width="96" height="96" alt="Cognisec AI Logo">
  <br>
  <strong>Cognisec AI — Enterprise Research Dashboard</strong>
</p>

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
| **Known Attack PR-AUC (Exp A)** | **0.9999** | **0.9432** | VAE achieves high PR-AUC under log1p variance stabilization |
| **Known Attack F1-Score (Exp A)** | **0.9997** [0.9996, 0.9998] | **0.4099** [0.4060, 0.4139] | $\Delta F1 = 0.5898$ ($p < 0.001$, Significant: True) |
| **Known Attack Recall (Exp A)** | **99.99%** | **25.89%** | Enforced under 1.0% benign false-alarm constraint |
| **Known Attack FPR (Exp A)** | **0.09%** | **0.93%** | Strictly complies with <1.0% operational ceiling |
| **Held-Out (Zero-Day) Recall (Exp B)**| **62.50%** [0.6229, 0.6277] | **16.60%** [0.1638, 0.1677] | $\Delta Recall = 0.4591$ ($p < 0.001$, Significant: True) |
| **Infiltration Attack Recall (Exp B)**| 0.00% | **47.22%** | **VAE detects stealth attacks completely missed by XGBoost** |
| **Inference Latency** | **0.00049 ms/sample** | 0.00522 ms/sample | XGBoost is ~10.6x faster on CPU |

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
# Step 1: Run Automated Test Suite (All 16 Tests Passing)
pytest tests/

# Step 2: Acquire Authentic CICIDS2017 Dataset
python -m src.data.download_cicids2017

# Step 3: Run Preprocessing & Leakage-Controlled Partitioning
python -m src.data.prepare_data

# Step 4: Execute Master Research Experiments (Exp A, B, C, D, F)
python -m src.experiments.run_experiments

# Step 5: Launch the Primary Web Application Dashboard (HTML5/CSS3/Vanilla JS)
python server.py
# or via CLI pipeline:
python main.py --step dashboard

# Optional: Launch Legacy Streamlit Demonstration Dashboard
python main.py --step streamlit
```

---

## 5. Web Application Architecture (Cognisec AI)

The primary user interface has been redesigned as an enterprise-grade cybersecurity research dashboard:

- **Frontend:** Semantic HTML5, CSS3 Custom Properties (enterprise light theme, deep navy typography, restrained blue/violet accents), Modular Vanilla JavaScript (ES6+), and Native High-DPI SVG Charts.
- **Backend API:** Lightweight, thread-safe Python HTTP layer (`server.py` / `src/api/server.py`) serving the REST endpoints:
  - `GET /` — Serves the unified web dashboard
  - `GET /api/health` — Live model and preprocessor status, environment and threshold verification
  - `GET /api/metadata` — CICIDS2017 flow counts, 70 retained features, duplicate statistics, and audit disclosures
  - `GET /api/experiments` — Raw results across Experiments A, B, C, D, and F
  - `GET /api/results/comparison` — Formatted comparative metrics with 95% bootstrap CIs and p-values
  - `GET /api/sample-csv` — 1-click authentic CICIDS2017 sample flow provider (100 flows)
  - `POST /api/analyze/csv-preview` — CSV schema validation, null/infinity audits, and 70-feature compatibility checking
  - `POST /api/analyze/xgboost` — Supervised flow inference with calibrated 0.52 classification threshold
  - `POST /api/analyze/vae` — PyTorch Tabular VAE reconstruction anomaly scoring with 0.3306 calibrated threshold
  - `POST /api/analyze/dual` — Joint comparative inference with agreement categorization (Both Flagged, XGBoost Only, VAE Only, Both Normal)

### Application Layout Sections:
1. **Overview:** Executive summary, KPI stat counters, model status cards, research problem statement, and key findings.
2. **Traffic Analysis:** Drag-and-drop CSV uploader, 1-click sample data loader, schema compatibility check against 70 flow features, missing-value audit, and interactive row preview.
3. **Attack Detection:** Dual-model inference engine, flow-by-flow probability and anomaly scoring, searchable and filterable prediction table, and CSV prediction export.
4. **Model Comparison:** Empirical comparison table (PR-AUC, ROC-AUC, F1, Recall, FPR, Latency), native SVG charts for per-family recall, threshold sensitivity (Exp C), distribution shift (Exp D), and authentic figure gallery.
5. **Unseen Attack Evaluation:** Dedicated zero-day research view with side-by-side held-out family disaggregation (DDoS, Web Attack, Bot, Infiltration), population caution badges, and complementarity donut chart (Exp F).
6. **Dataset & Experiments:** Partitioning manifest breakdown (60% Train, 20% Val, 20% Test), leakage prevention controls, and class distribution table.
7. **Research Methodology:** 8-stage experimental pipeline diagram, and transparent disclosure of the 14,682 identical feature vectors crossing train/test boundaries for known attacks.
8. **System Information:** Live API health monitor, artifact statuses, framework versions (Python, PyTorch, XGBoost), and execution environment.

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
#   C o g n i s e c _ A I  
 