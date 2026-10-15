# Cognisec AI: Project Cleanup and Submission Optimization Final Report

**Project:** Cognisec AI: Network Behaviour Fingerprinting for Unseen Attack Detection — An Experimental Comparison of XGBoost and Variational Autoencoders  
**Document:** `reports/cleanup_final_report.md`  
**Phase:** Stage E — Final Execution & Verification Report  
**Date:** October 16, 2026  
**Auditor / Developer:** Santhoskrishna / Cognisec AI Team  
**Status:** **CLEANUP COMPLETE & VERIFIED 100% OPERATIONAL**  

---

## 1. Executive Summary

A controlled, non-destructive repository cleanup was executed in strict adherence to project safety rules and the user-approved cleanup plan ([`reports/cleanup_manifest.md`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/reports/cleanup_manifest.md)).

All regenerable setuptools build caches, pytest runtime caches, compiled Python bytecode directories, and redundant pre-fix temporary backups were purged. 100% of machine learning model checkpoints, raw and processed datasets, scientific reports, academic figures, submission evidence screenshots, top-level entry point scripts, and the Python virtual environment (`.venv/`) were preserved intact.

Post-cleanup regression testing confirmed that the entire 32-test automated suite passes cleanly, the Flask application serves all assets and endpoints, and dual-model inference operates without error.

---

## 2. Inventory of Deleted Files and Directories

| Path | Type | Original Size (Bytes) | Justification & Verification |
| :--- | :---: | :---: | :--- |
| `build/` | Directory (29 files) | 129,419 B (126.4 KB) | Ephemeral setuptools build directory. Regenerable on demand via `pip install -e .`. |
| `.pytest_cache/` | Directory (5 files) | 2,527 B (2.5 KB) | Ephemeral pytest execution cache. Re-created automatically on next test run. |
| `__pycache__/` (root) | Directory | ~4,000 B | Root-level compiled Python bytecode. Regenerable on demand. |
| `src/**/__pycache__/` | 8 Directories (47 files) | 548,774 B (535.9 KB) | Bytecode cache across `src/` modules. Regenerated automatically by Python runtime. |
| `tests/**/__pycache__/` | Directory | ~8,000 B | Bytecode cache for test fixtures. Regenerated automatically. |
| `src/live/flow_extractor.py.bak_stage_c` | Single File | 22,283 B (21.8 KB) | Pre-fix backup created prior to Stage C. Verified permanently preserved in Git commit `audit-baseline` and remote commit `22fa560`. |
| **Total Items Deleted** | **13 Directories + 1 File** | **~715,003 B (~0.68 MB)** | **Zero impact on codebase, models, or documentation.** |

---

## 3. Files Moved

**None.** All files currently reside in their designated architectural locations.

---

## 4. Files Retained (Preservation Audit)

### A. Academic & Submission Screenshots (100% Retained)
All 6 screenshots were verified as active submission evidence and preserved in [`reports/screenshots/`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/reports/screenshots/):
- `01_dashboard.png` (161.4 KB): Full Web Dashboard interface.
- `02_csv_upload.png` (91.2 KB): CSV Batch Preview interface.
- `03_xgboost_predictions.png` (122.5 KB): XGBoost classification table.
- `04_vae_anomaly_scores.png` (129.7 KB): VAE anomaly scoring view.
- `05_model_comparison.png` (148.2 KB): Dual-model comparison metrics.
- `06_validation_error.png` (100.6 KB): Validation error handling.

### B. Research Figures (100% Retained)
All 8 publication-grade research figures were verified and preserved in [`reports/figures/`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/reports/figures/):
- `fig1_dataset_distribution.png`: Class balance chart.
- `fig2_roc_pr_curves.png`: Comparative ROC and PR curves.
- `fig3_confusion_matrices.png`: Dual confusion matrices.
- `fig4_anomaly_score_distributions.png`: Benign vs Attack MSE distributions.
- `fig5_per_family_recall.png`: Per-attack-family breakdown.
- `fig6_threshold_sensitivity.png`: Decision threshold sensitivity.
- `fig7_distribution_shift_degradation.png`: Generalization degradation curve.
- `fig8_model_disagreement.png`: Model complementarity Venn/disagreement analysis.

### C. Top-Level Scripts & Batch Launchers (100% Retained)
Every script was verified to serve an active operational or documentation purpose:
- [`server.py`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/server.py): Flask entrypoint on `http://127.0.0.1:5000`.
- [`main.py`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/main.py): Unified CLI orchestrator for pipeline tasks.
- [`app.py`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/app.py): Streamlit demonstration dashboard.
- [`capture_screenshots.py`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/capture_screenshots.py): Automated browser screenshot capture utility.
- [`generate_pptx.py`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/generate_pptx.py): Programmatic PowerPoint viva compiler.
- [`sync_submission_package.py`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/sync_submission_package.py): Automated submission package packager.
- [`run_dashboard.bat`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/run_dashboard.bat): 1-click launcher for Flask dashboard.
- [`run_streamlit.bat`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/run_streamlit.bat): 1-click launcher for Streamlit app.
- [`run_tests.bat`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/run_tests.bat): 1-click batch test runner.
- [`m.txt`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/m.txt): MSc Master Research Prompt Specification.

### D. Model Checkpoints & Preprocessors (100% Retained)
- [`artifacts/models/xgboost_model.json`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/artifacts/models/xgboost_model.json)
- [`artifacts/models/vae_checkpoint.pt`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/artifacts/models/vae_checkpoint.pt)
- [`artifacts/models/flow_preprocessor.joblib`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/artifacts/models/flow_preprocessor.joblib)

### E. Virtual Environment
- `.venv/` (Python 3.12.10 environment with PyTorch, Scapy, XGBoost, Flask) was preserved untouched.

---

## 5. Files Backed Up

- `src/live/flow_extractor.py.bak_stage_c`: Permanently committed and preserved in the repository Git history (Git commit `audit-baseline` and remote commit `22fa560` on GitHub).
- All source files and reports are backed up on GitHub remote [`https://github.com/SanthoskrishnaG/Cognisec_AI.git`](https://github.com/SanthoskrishnaG/Cognisec_AI.git).

---

## 6. Disk Space Recovered

- **Total Safe Space Recovered:** **~715,003 Bytes (~0.68 MB)**
- **Working Tree State:** Clean and free of stale build directories and temporary backup files.

---

## 7. Post-Cleanup Verification & Test Results

### 1. Full Automated Pytest Suite
Executed command:
`.\.venv\Scripts\python.exe -m pytest -q`
```
................................                                         [100%]
32 passed in 11.49s
```
- **Exit Code:** `0`
- **Total Tests:** 32
- **Passed:** 32 (100%)
- **Failed:** 0
- **Skipped:** 0

### 2. End-to-End Live HTTP Server Verification
Executed command:
`.\.venv\Scripts\python.exe tests/verify_live_server.py`
```
[1] Testing GET / -> [OK] Index HTML rendered with all 8 sections.
[2] Testing static assets -> [OK] CSS, JS modules, and figures served.
[3] Testing API health -> [OK] System health: healthy, XGBoost threshold: 0.52, VAE threshold: 0.330609858
[4] Testing API sample-csv -> [OK] Sample CSV fetched (100 rows).
[5] Testing POST /api/analyze/csv-preview -> [OK] CSV Preview compatible: 70/70 features matched.
[6] Testing POST /api/analyze/xgboost -> [OK] XGBoost evaluated 100 flows: 0 attacks, 100 benign.
[7] Testing POST /api/analyze/vae -> [OK] VAE evaluated 100 flows: 2 anomalies, 98 normal.
[8] Testing POST /api/analyze/dual -> [OK] Dual Inference Summary: 98 Normal, 2 VAE Only.
[9] Testing API comparison & metadata -> [OK] Loaded.
[10] Testing Live Monitoring Module & Endpoints:
    [OK] Live interfaces enumerated (48 interfaces detected).
    [OK] Live status telemetry verified.
    [OK] Live capture start/stop lifecycle confirmed.
    [OK] Live flows query returned 5 flows.

>>> ALL LIVE HTTP SERVER ENDPOINTS AND INFERENCE PIPELINES VERIFIED 100% SUCCESSFUL! <<<
```

### 3. Submission & Research Assets Verification
Checked 27 core submission files, research reports, models, screenshots, and presentation slide decks:
```
ALL 27 SUBMISSION AND RESEARCH ASSETS VERIFIED PRESENT AND NON-EMPTY!
```

---

## 8. Remaining Cleanup Recommendations

1. **Keep `.gitignore` Maintained:**
   The updated [`.gitignore`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/.gitignore) already includes `build/`, `.pytest_cache/`, `__pycache__/`, `*.log`, and `*.bak*` patterns to ensure future builds or tests do not leave untracked clutter.
2. **Commit Cleanup Artifacts:**
   Stage and commit [`reports/cleanup_manifest.md`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/reports/cleanup_manifest.md) and [`reports/cleanup_final_report.md`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/reports/cleanup_final_report.md) to finalize the audit documentation trail.
