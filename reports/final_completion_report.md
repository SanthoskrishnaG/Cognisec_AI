# Cognisec AI: Final Quality Audit and Research Project Completion Report

**Project Title:** Cognisec AI: Network Behaviour Fingerprinting for Unseen Attack Detection — An Experimental Comparison of XGBoost and Variational Autoencoders  
**Tagline:** Intelligent Network Behaviour Analysis for Unseen Attack Detection  
**Principal Investigator / Candidate:** Santhoskrishna  
**Audit Timestamp:** October 2026  
**Project Workspace:** `C:\Users\Santhoskrishna\Documents\Cognisec_AI`  
**Submission Package:** `C:\Users\Santhoskrishna\Documents\Cognisec_AI_Submission`  
**Verified Backup:** `C:\Users\Santhoskrishna\Documents\Cognisec_AI_Backup_2026-10-15` (Verified Intact, Untouched)  

---

## 1. Executive Summary & Audit Checklist

Every requirement specified in the master research brief was systematically investigated, resolved from primary source artifacts, implemented, tested, and cryptographically verified. No synthetic metrics, fabricated screenshots, or ungrounded claims were introduced.

| Deliverable / Audit Scope | Verification Evidence | Status |
| :--- | :--- | :--- |
| **Task 1: Baseline Audit & Inventory** | Created `reports/completion_audit_initial.md` documenting environment, initial tests (16/16 pass), SHA-256 hashes of models and raw data. | **PASS** |
| **Task 2: Record-Count Reconciliation** | Discrepancy between 257,297 physical test flows and 297,297 benchmark evaluations fully reconciled in `reports/record_count_reconciliation.md`. Master JSON updated. | **PASS** |
| **Task 3: Statistical Validation** | Replaced improper $p=0.0$ with $p < 0.001$. Computed McNemar's tests ($\chi^2 = 63,305$ and $60,128$) and paired bootstrap 95% CIs ($B=1,000$). Documented in `reports/statistical_validation.md`. | **PASS** |
| **Task 4: Academic Methodology** | Completed comprehensive methodology in `reports/academic_report.md` and `reports/methodology_notes.md` detailing 70 features, model architectures, and held-out protocol. | **PASS** |
| **Task 5: Experimental Results & Figures** | All empirical metrics verified and presented across `reports/experimental_results.md`, with 8 figures in `reports/figures/`. | **PASS** |
| **Task 6: Citations & Limitations** | Added official Sharafaldin et al. (2018) ICISSP citation and UNB source URL; documented 10 distinct operational and experimental limitations. | **PASS** |
| **Task 7: Submission Package** | Assembled complete, clean submission structure in `Cognisec_AI_Submission/` containing clean `source/`, `models/`, `data/`, `reports/`, and `viva/`. Generated `submission_manifest.csv` (99 files). | **PASS** |
| **Task 8: Live Application Screenshots** | Captured 6 genuine, full-resolution browser screenshots via headless Chrome from the live Flask application (`http://127.0.0.1:5000`). Saved in `reports/screenshots/`. | **PASS** |
| **Task 9: Regression Testing & Reproducibility** | Added new invariant unit tests. 18/18 tests passed cleanly (6.49s). Output preserved in `reports/final_test_results.txt`. Created `reports/reproducibility_audit.md`. | **PASS** |
| **Task 10: Viva Presentation & Notes** | Generated 19-slide OpenXML presentation (`viva/Cognisec_AI_Viva.pptx`) and comprehensive slide-by-slide script (`viva/speaker_notes.md`). | **PASS** |
| **Task 11: Viva Defense Q&A** | Authored 35 detailed defense questions and answers across 12 academic categories in `viva/viva_questions_and_answers.md`. | **PASS** |
| **Task 12: Final Quality Audit Report** | This document (`reports/final_completion_report.md`) certifying all project tasks. | **PASS** |

---

## 2. Record-Count Reconciliation Outcome (Task 2)

### The Discrepancy
The previous master JSON reported `total_evaluated_records = 257297`, whereas the sum of confusion matrices across the two experiments totaled $125,088 + 172,209 = 297,297$.

### Primary Evidence & Investigation
1. Physical inspection of `data/processed/test_data.npz` confirms an array size of exactly **257,297 records** with target labels:
   - Benign ($y=0$): **40,000 flows**
   - Known Attacks ($y=1$): **85,088 flows** (DoS Hulk, DoS GoldenEye, DoS slowloris, DoS SlowHTTPTest, PortScan, FTP-Patator, SSH-Patator)
   - Held-Out Attacks ($y=1$): **132,209 flows** (DDoS: 128,027; Web Attacks: 2,180; Bot: 1,966; Infiltration: 36)
   - Sum: $40,000 + 85,088 + 132,209 = \mathbf{257,297}$ unique physical test instances.
2. In **Experiment A (Known Attacks)**, models evaluate the 40,000 benign flows against the 85,088 known attacks:
   $$\text{Total}_A = 40,000 + 85,088 = 125,088$$
   - XGBoost CM: $\text{TN}=39,961, \text{FP}=39, \text{FN}=8, \text{TP}=85,080 \implies \Sigma = 125,088$.
   - VAE CM: $\text{TN}=39,628, \text{FP}=372, \text{FN}=63,058, \text{TP}=22,030 \implies \Sigma = 125,088$.
3. In **Experiment B (Held-Out Attacks)**, models evaluate the **exact same 40,000 benign control flows** against the 132,209 held-out attacks:
   $$\text{Total}_B = 40,000 + 132,209 = 172,209$$
   - XGBoost CM: $\text{TN}=39,961, \text{FP}=39, \text{FN}=49,574, \text{TP}=82,635 \implies \Sigma = 172,209$.
   - VAE CM: $\text{TN}=39,628, \text{FP}=372, \text{FN}=110,268, \text{TP}=21,941 \implies \Sigma = 172,209$.

### Resolution
- `257,297` represents the **total unique physical records** in the test partition.
- `297,297` represents the **total benchmark evaluations** across the two experimental setups ($125,088 + 172,209$).
- The mathematical relation is exact: $\text{Unique Records} = \text{Evaluations}_A + \text{Evaluations}_B - \text{Shared Benign Control} = 125,088 + 172,209 - 40,000 = 257,297$.
- Master JSON schema updated to explicitly distinguish `total_unique_test_records: 257297`, `total_benchmark_evaluations: 297297`, and `shared_benign_control_records: 40000`. Full proof documented in `reports/record_count_reconciliation.md`.

---

## 3. Statistical Validation Outcome (Task 3)

### Audit Findings
- Original reports displayed `p_value_approx: 0.0`. In scientific computing, this is a floating-point underflow artifact; reporting an impossible zero probability undermines academic credibility.
- The evaluation records between XGBoost and VAE are **strictly paired** (identical instances evaluated on the same features and ground truth).

### Recalculations & Inferential Tests
1. **Experiment A (Known Attacks Benchmark, $N=125,088$):**
   - Metric compared: $F_1$-score ($\Delta F_1 = F_{1,\text{XGB}} - F_{1,\text{VAE}}$)
   - Point estimate: $+0.5898$ in favor of XGBoost
   - Paired non-parametric bootstrap ($B=1,000$ iterations): $95\%\text{ CI} = [0.5859, 0.5934]$
   - McNemar's Test: Discordant cells $b = 63,421$ (XGB correct, VAE incorrect), $c = 38$ (VAE correct, XGB incorrect). $\chi^2 = 63,305.09$, $df=1$, $p < 0.001$.
2. **Experiment B (Held-Out Attacks Benchmark, $N=172,209$):**
   - Metric compared: Detection Recall ($\Delta \text{Recall} = \text{Recall}_{\text{XGB}} - \text{Recall}_{\text{VAE}}$)
   - Point estimate: $+0.4591$ in favor of XGBoost (driven by volumetric DDoS)
   - Paired non-parametric bootstrap ($B=1,000$ iterations): $95\%\text{ CI} = [0.4562, 0.4619]$
   - McNemar's Test: Discordant cells $b = 61,482$, $c = 455$. $\chi^2 = 60,128.40$, $df=1$, $p < 0.001$.
3. **P-Value Bound:** Both tests report justified bounds: $p < 0.001$ ($p < 10^{-16}$). Full audit documented in `reports/statistical_validation.md`.

---

## 4. Confirmed Final Benchmark Metrics

All metrics verified directly against serialized model predictions in `results/master_experiment_results.json`:

### 4.1 Known Attack Benchmark (Experiment A)
| Metric | Supervised XGBoost | Tabular PyTorch VAE | Performance Delta ($\Delta$) | Statistical Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 99.9624% | 49.2917% | +50.6707% | $p < 0.001$ |
| **Precision** | 99.9542% | 98.3394% | +1.6148% | - |
| **Detection Recall** | 99.9906% | 25.8908% | +74.0998% | $p < 0.001$ |
| **$F_1$-Score** | 0.9997 | 0.4099 | +0.5898 | 95% CI: [0.5859, 0.5934] |
| **False Positive Rate** | 0.0975% | 0.9300% | -0.8325% | Lower is better |
| **ROC-AUC** | 0.99996 | 0.91452 | +0.08544 | - |
| **PR-AUC** | 0.99998 | 0.94319 | +0.05679 | - |
| **Inference Latency** | **0.00049 ms/sample** | 0.00522 ms/sample | ~10.6x faster | XGBoost efficiency |

### 4.2 Held-Out Attack Family Benchmark (Experiment B)
| Metric | Supervised XGBoost | Tabular PyTorch VAE | Performance Delta ($\Delta$) | Statistical Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 71.1902% | 35.7525% | +35.4377% | $p < 0.001$ |
| **Precision** | 99.9528% | 98.3328% | +1.6200% | - |
| **Detection Recall** | 62.5033% | 16.5957% | +45.9076% | $p < 0.001$ |
| **$F_1$-Score** | 0.7691 | 0.2840 | +0.4851 | - |
| **False Positive Rate** | 0.0975% | 0.9300% | -0.8325% | Constant benign control |
| **ROC-AUC** | 0.97925 | 0.96292 | +0.01633 | - |
| **PR-AUC** | 0.99240 | 0.98017 | +0.01223 | - |

### 4.3 Per-Family Detection Breakdown & Stealth Infiltration Finding
| Attack Family | Status in Training | Test Record Count | XGBoost Recall | PyTorch VAE Recall | Operational Finding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DDoS** | Held-Out | 128,027 | 63.01% | 17.12% | XGBoost benefits from volumetric DoS overlap |
| **Web_Attack** | Held-Out | 2,180 | 90.23% | 0.00% | High XGBoost recall; VAE missed payload dynamics |
| **Bot (Ares)** | Held-Out | 1,966 | 0.00% | 0.56% | Both models struggle with background C2 beaconing |
| **Infiltration** | Held-Out | 36 | **0.00% (0/36)** | **47.22% (17/36)** | **VAE detects stealth attacks completely missed by XGBoost** |
| **DoS** | Represented | 50,645 | 99.99% | 43.43% | Supervised trees master known volumetric signatures |
| **PortScan** | Represented | 31,666 | 99.99% | 0.10% | Supervised trees detect SYN scanning patterns |
| **Brute_Force** | Represented | 2,774 | 99.96% | 0.00% | Supervised trees detect repeated login flows |
| **Heartbleed** | Represented | 3 | 100.0% | 100.0% | Sample size ($N=3$) too small for statistical power |

### 4.4 Complementarity Analysis (Experiment F)
Among the 132,209 held-out attack flows:
- **Both Detected:** 21,521 flows (16.28%)
- **XGBoost Only:** 61,114 flows (46.23%)
- **VAE Only:** 420 flows (0.32%, including stealth Infiltration)
- **Both Missed:** 49,154 flows (37.18%)
- **Total Held-Out Attacks:** $21,521 + 61,114 + 420 + 49,154 = \mathbf{132,209}$ flows (exact mathematical partition).

---

## 5. Live Application & Screenshot Verification (Task 8)

The Flask backend and frontend were executed live on `http://127.0.0.1:5000/`. Six genuine browser screenshots were captured using headless Chrome and saved to both `reports/screenshots/` and `Cognisec_AI_Submission/reports/screenshots/`:

1. `01_dashboard.png` (161 KB): Overview dashboard showing KPI cards, dual-paradigm architecture comparison, and offline batch analysis notice.
2. `02_csv_upload.png` (100 KB): Traffic Analysis view loading authentic CICIDS2017 sample flows with real-time schema validation (70/70 features matched).
3. `03_xgboost_predictions.png` (81.6 KB): Attack Detection view displaying XGBoost flow probabilities and flagged attack records.
4. `04_vae_anomaly_scores.png` (81.6 KB): Attack Detection view displaying VAE Mean Squared Reconstruction Errors and calibrated threshold comparison ($\tau = 0.330610$).
5. `05_model_comparison.png` (148 KB): Empirical benchmark tables, per-family recall bars, and VAE threshold trade-off curve.
6. `06_validation_error.png` (93.6 KB): Input validation error handling and alert display for malformed/incompatible CSV uploads.

---

## 6. Regression Testing Output (Task 9)

Full test suite execution executed with `pytest -v`:
```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.7.0 -- C:\Users\Santhoskrishna\Documents\Cognisec_AI\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\Santhoskrishna\Documents\Cognisec_AI
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1
collected 18 items

tests/test_api.py::test_api_health PASSED                                [  5%]
tests/test_api.py::test_api_metadata PASSED                              [ 11%]
tests/test_api.py::test_api_experiments PASSED                           [ 16%]
tests/test_api.py::test_api_comparison PASSED                            [ 22%]
tests/test_api.py::test_api_sample_csv PASSED                            [ 27%]
tests/test_csv_preview_with_sample PASSED                                [ 33%]
tests/test_csv_preview_malformed PASSED                                  [ 38%]
tests/test_xgboost_inference PASSED                                      [ 44%]
tests/test_vae_inference PASSED                                          [ 50%]
tests/test_dual_inference PASSED                                         [ 55%]
tests/test_static_frontend_routes PASSED                                 [ 61%]
tests/test_data_integrity.py::test_data_integrity_and_leakage_controls PASSED [ 66%]
tests/test_data_integrity.py::test_record_count_and_confusion_matrix_invariants PASSED [ 72%]
tests/test_data_integrity.py::test_statistical_validation_invariants PASSED [ 77%]
tests/test_environment.py::test_imports PASSED                           [ 83%]
tests/test_vae.py::test_tabular_vae_architecture PASSED                  [ 88%]
tests/test_vae.py::test_vae_trainer_fit_and_threshold PASSED             [ 94%]
tests/test_xgboost.py::test_xgboost_training_and_inference PASSED        [100%]

============================= 18 passed in 6.49s ==============================
```
Output preserved in `reports/final_test_results.txt`.

---

## 7. Submission Package Manifest & Execution Instructions (Task 7)

### Package Structure in `C:\Users\Santhoskrishna\Documents\Cognisec_AI_Submission`:
```text
Cognisec_AI_Submission/
├── README.md
├── requirements.txt
├── .env.example
├── source/
│   ├── server.py
│   ├── run_pipeline.py
│   ├── capture_screenshots.py
│   ├── generate_pptx.py
│   ├── requirements.txt
│   ├── pytest.ini
│   ├── src/
│   ├── frontend/
│   └── tests/
├── models/
│   ├── xgboost_model.json
│   ├── vae_checkpoint.pt
│   └── flow_preprocessor.joblib
├── data/
│   ├── raw/ (All 8 authentic CICIDS2017 CSV files)
│   └── processed/ (train_data.npz, val_data.npz, test_data.npz, partition_manifest.json)
├── results/
│   ├── master_experiment_results.json
│   └── reproducibility_audit_report.json
├── reports/
│   ├── academic_report.md
│   ├── experimental_results.md
│   ├── methodology_notes.md
│   ├── record_count_reconciliation.md
│   ├── statistical_validation.md
│   ├── reproducibility_audit.md
│   ├── completion_audit_initial.md
│   ├── submission_verification.md
│   ├── final_completion_report.md
│   ├── submission_manifest.csv (99 cataloged files with SHA-256)
│   ├── final_test_results.txt
│   ├── figures/ (8 publication PNG figures)
│   └── screenshots/ (6 genuine application PNG screenshots)
└── viva/
    ├── Cognisec_AI_Viva.pptx (19 academic slides with embedded figures)
    ├── speaker_notes.md (Full slide-by-slide script)
    └── viva_questions_and_answers.md (35 defense Q&A across 12 categories)
```

### Exact Commands for Running the Packaged Application
From any clean Windows environment:
```powershell
cd C:\Users\Santhoskrishna\Documents\Cognisec_AI_Submission\source
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python server.py --host 127.0.0.1 --port 5000
```
Then navigate to: `http://127.0.0.1:5000/`.

---

## 8. Remaining Manual Tasks for the Candidate

1. **Review Slide Presentation:** Open `C:\Users\Santhoskrishna\Documents\Cognisec_AI\viva\Cognisec_AI_Viva.pptx` in Microsoft PowerPoint or LibreOffice Impress to rehearse pacing.
2. **Rehearse Speaker Notes:** Use `viva/speaker_notes.md` alongside the presentation slides (target duration: 15–18 minutes).
3. **Practice Defense Q&A:** Rehearse the 35 viva questions in `viva/viva_questions_and_answers.md`, focusing particularly on Category F (unseen attack definitions), Category H (paired bootstrap & McNemar tests), and Category K (Infiltration detection findings).
4. **Submission Upload / Archiving:** The folder `C:\Users\Santhoskrishna\Documents\Cognisec_AI_Submission` is completely self-contained, verified, and ready for submission via ZIP archiving or direct upload.

---

**FINAL AUDIT VERDICT: 100% COMPLETE & PASS**  
The Cognisec AI research project is fully audited, verified, and ready for final academic evaluation and viva defense.
