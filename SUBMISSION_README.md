# Cognisec AI: Academic Research Submission Package

**Project Title:** Cognisec AI: Network Behaviour Fingerprinting for Unseen Attack Detection — An Experimental Comparison of XGBoost and Variational Autoencoders  
**Product Name:** Cognisec AI  
**Tagline:** Intelligent Network Behaviour Analysis for Unseen Attack Detection  
**Candidate Name:** Santhoskrishna  
**Academic Degree:** Advanced MSc Data Science / Cyber Security  
**Submission Package Location:** `C:\Users\Santhoskrishna\Documents\Cognisec_AI_Submission`  
**Verified Backup:** `C:\Users\Santhoskrishna\Documents\Cognisec_AI_Backup_2026-10-15` (Verified Intact)  
**Submission Date:** October 2026  

---

## 1. Directory Structure

```text
Cognisec_AI_Submission/
├── README.md                          # Project documentation and architecture guide
├── SUBMISSION_README.md               # Academic submission metadata and defense guide
├── requirements.txt                   # Verified Python 3.12 dependency manifest
├── .env.example                       # Sanitized environment configuration template
│
├── source/                            # Complete, standalone application source
│   ├── server.py                      # Flask REST API server entry point
│   ├── main.py                        # Pipeline execution entry point
│   ├── app.py                         # Streamlit UI (legacy dashboard)
│   ├── capture_screenshots.py         # Automated UI screenshot verification script
│   ├── generate_pptx.py               # OpenXML viva presentation generator
│   ├── run_dashboard.bat              # One-click Windows runner for Flask web app
│   ├── run_streamlit.bat              # Runner for Streamlit app
│   ├── run_tests.bat                  # One-click Windows pytest runner
│   ├── requirements.txt               # Source-level dependency manifest
│   ├── pytest.ini                     # Pytest configuration
│   ├── src/                           # Modular Python packages
│   │   ├── api/                       # Flask controllers and REST routing
│   │   ├── models/                    # XGBoost wrapper and Tabular PyTorch VAE
│   │   ├── preprocessing/             # 70-feature pipeline and leakage filters
│   │   ├── evaluation/                # Metrics, McNemar tests, bootstrap CIs
│   │   └── utils/                     # Logging, serialization, helpers
│   ├── frontend/                      # Responsive HTML5, CSS3, vanilla ES6 JS
│   │   ├── index.html                 # Research workbench user interface
│   │   ├── css/styles.css             # Design tokens, themes, layout
│   │   ├── js/ (app.js, api.js, charts.js) # Vanilla JS controllers
│   │   └── assets/                    # SVG logos and icons
│   ├── tests/                         # Pytest test suite (18/18 passing)
│   └── configs/                       # Hyperparameter and experiment YAML configs
│
├── models/                            # Serialized model artifacts
│   ├── xgboost_model.json             # Trained supervised binary tree ensemble
│   ├── vae_checkpoint.pt              # Trained PyTorch Tabular VAE (benign manifold)
│   └── flow_preprocessor.joblib       # Fitted RobustScaler (70 behavioral features)
│
├── data/                              # Datasets and processed arrays
│   ├── raw/                           # All 8 authentic CICIDS2017 flow CSV files
│   └── processed/                     # train_data.npz, val_data.npz, test_data.npz, partition_manifest.json
│
├── results/                           # Verified experimental results
│   ├── master_experiment_results.json # Final benchmark JSON (257k unique / 297k evals)
│   └── reproducibility_audit_report.json # Bit-for-bit reproduction report
│
├── reports/                           # Academic reports, statistics & figures
│   ├── academic_report.md             # Complete academic thesis / paper
│   ├── experimental_results.md        # Comprehensive benchmark tables & findings
│   ├── methodology_notes.md           # Standalone mathematical methodology
│   ├── record_count_reconciliation.md # Resolution of 257,297 vs. 297,297 counts
│   ├── statistical_validation.md      # McNemar tests and paired bootstrap 95% CIs
│   ├── reproducibility_audit.md       # Reproducibility certificate and commands
│   ├── submission_manifest.csv        # Cryptographic SHA-256 manifest (all files)
│   ├── submission_verification.md     # Package verification audit
│   ├── final_completion_report.md     # Final Quality Audit report (All PASS)
│   ├── final_test_results.txt         # Complete output of 18 passing tests
│   ├── figures/                       # 8 high-resolution publication PNG charts
│   └── screenshots/                   # 6 genuine application browser screenshots
│
└── viva/                              # Viva defense preparation materials
    ├── Cognisec_AI_Viva.pptx          # 19-slide OpenXML presentation with visuals
    ├── speaker_notes.md               # Slide-by-slide defense script
    └── viva_questions_and_answers.md  # 35 defense questions across 12 categories
```

---

## 2. Quick-Start Execution Guide

### Prerequisites
- Windows 10/11 (or Linux/macOS)
- Python 3.10 to 3.12 (Python 3.12.10 recommended)

### One-Click Startup (Windows)
Double-click `source/run_dashboard.bat`, or execute via PowerShell:
```powershell
cd C:\Users\Santhoskrishna\Documents\Cognisec_AI_Submission\source
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python server.py --host 127.0.0.1 --port 5000
```
Open your browser at: `http://127.0.0.1:5000/`.

### Running Automated Verification Tests
```powershell
.\.venv\Scripts\activate
python -m pytest -v
```
Expected output: **18 passed in ~6.5s**.

---

## 3. Academic Defense Highlights

1. **Two Opposing Paradigms:** Supervised tree boosting (XGBoost) vs. Unsupervised latent density modeling (PyTorch Tabular VAE).
2. **Leakage Elimination:** 8 socket-identity features purged; 70 behavioral features retained; zero train-test identity contamination.
3. **Reconciled Sample Counts:** Exactly 257,297 unique physical test flows yield 297,297 benchmark evaluations across Experiments A and B, sharing a 40,000 benign control baseline.
4. **Key Finding:** While XGBoost dominates high-volume known threats, it scored **0.0% recall** on stealthy Infiltration attacks. The VAE successfully caught **47.2%** of Infiltration flows without prior attack training.
5. **Architectural Recommendation:** A hybrid multi-tiered deployment (XGBoost for perimeter throughput $\to$ VAE for zero-day anomaly triage).

---

## 4. Integrity and Compliance Statement
- **Backup Intact:** The verified backup at `C:\Users\Santhoskrishna\Documents\Cognisec_AI_Backup_2026-10-15` has not been altered or deleted.
- **No Fabricated Data:** All metrics, confusion matrices, p-value bounds ($p < 0.001$), figures, and screenshots derive directly from real models and authentic CICIDS2017 data.
- **Sanitized Secrets:** No credentials, API tokens, or private paths are exposed.
