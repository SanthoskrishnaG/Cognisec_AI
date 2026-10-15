# Cognisec AI: Project Cleanup and Submission Optimization Manifest

**Project:** Cognisec AI: Network Behaviour Fingerprinting for Unseen Attack Detection  
**Document:** `reports/cleanup_manifest.md`  
**Phase:** Stage A — Comprehensive Inventory & Risk Assessment  
**Date:** October 16, 2026  
**Auditor:** Santhoskrishna / Cognisec AI Team  
**Status:** **INVENTORY COMPLETE (Stage A) — AWAITING USER APPROVAL (Stage B Review)**  

---

## 1. Safety Principles & Cleanup Scope

In accordance with strict research integrity and repository preservation rules:
1. **Zero Model Modifications:** Trained artifacts ([`xgboost_model.json`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/artifacts/models/xgboost_model.json), [`vae_checkpoint.pt`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/artifacts/models/vae_checkpoint.pt), [`flow_preprocessor.joblib`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/artifacts/models/flow_preprocessor.joblib)) are strictly preserved.
2. **Zero Scientific Data Loss:** Raw datasets, processed partitions ($N=151,530$), statistical validation reports, and academic research logs are preserved.
3. **Environment Preservation:** The local virtual environment (`.venv/`) is strictly retained and never deleted.
4. **No Destructive Action in Stage A:** No files have been deleted. Deletions will only occur in Stage C upon explicit user approval.
5. **Pre-Deletion Backup Requirement:** Any approved deletions will be backed up or verified against Git before removal.

---

## 2. Categorized File Inventory

### Category 1: `SAFE_TO_DELETE` (Regenerable Caches & Build Artifacts)

These items are temporary build outputs, bytecode caches, or test caches that can be regenerated on demand. None contain original source code or research evidence.

| Path | File/Dir Type | Size (Bytes) | References Checked | Reason & Justification | Backup / Git Status | Risk Level |
| :--- | :---: | :---: | :--- | :--- | :--- | :---: |
| `build/` | Directory (29 files) | 129,419 B (126.4 KB) | Entire workspace | Ephemeral build directory created by setuptools. Already listed in `.gitignore`. | Completely regenerable via `pip install -e .` | **NONE** |
| `.pytest_cache/` | Directory (5 files) | 2,527 B (2.5 KB) | Pytest internal | Pytest execution cache. Already listed in `.gitignore`. | Completely regenerable via `pytest` | **NONE** |
| `__pycache__/` (root) | Directory | ~4,000 B | Python runtime | Python bytecode cache (`.pyc`). Already listed in `.gitignore`. | Completely regenerable by Python | **NONE** |
| `src/**/__pycache__/` | 8 Directories (47 files) | 548,774 B (535.9 KB) | Python runtime | Compiled Python bytecode across `src/` modules. | Completely regenerable by Python | **NONE** |
| `tests/**/__pycache__/` | Directory | ~8,000 B | Python runtime | Compiled Python bytecode for test files. | Completely regenerable by Python | **NONE** |
| `src/live/flow_extractor.py.bak_stage_c` | Single File | 22,283 B (21.8 KB) | Entire workspace | Pre-fix backup created before Stage C. Stage C fixes are now verified, committed, and pushed to GitHub (commit `22fa560`). | Permanent copy safely archived in Git commit `audit-baseline` and commit history. | **LOW** |

**Estimated Recoverable Space from `SAFE_TO_DELETE`:** **~707,003 Bytes (690.4 KB / 0.67 MB)**

---

### Category 2: `SAFE_TO_MOVE`

No files require moving at this time. All files currently reside in their designated architectural locations.

---

### Category 3: `KEEP_REQUIRED` (Application, Scripts & Entry Points)

Every top-level script and entry point was inspected line-by-line for active dependencies and architectural roles:

| File / Component | Size (Bytes) | Role & References | Decision | Justification |
| :--- | :---: | :--- | :---: | :--- |
| [`server.py`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/server.py) | 5,708 B | Flask entrypoint; referenced by `run_dashboard.bat`, `main.py`, `tests/verify_live_server.py`. | **KEEP_REQUIRED** | Primary entrypoint for the web application and REST API on `http://127.0.0.1:5000`. |
| [`main.py`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/main.py) | 8,362 B | Unified CLI orchestrator; referenced by `README.md`, `SUBMISSION_README.md`. | **KEEP_REQUIRED** | Dispatches all scientific pipeline steps (`--step download`, `prepare`, `train-vae`, `train-xgb`, `experiments`, `server`). |
| [`app.py`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/app.py) | 18,468 B | Streamlit research demo dashboard; referenced by `run_streamlit.bat`, `main.py`. | **KEEP_REQUIRED** | Interactive demonstration interface for academic presentation. |
| [`capture_screenshots.py`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/capture_screenshots.py) | 7,333 B | Playwright browser automation script; referenced by `SUBMISSION_README.md`. | **KEEP_REQUIRED** | Generates verifiable academic screenshot evidence for the submission package. |
| [`generate_pptx.py`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/generate_pptx.py) | 26,662 B | Automated PowerPoint compiler; generates `viva/Cognisec_AI_Viva.pptx`. | **KEEP_REQUIRED** | Programmatically compiles the 15-slide master viva presentation with real research metrics. |
| [`sync_submission_package.py`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/sync_submission_package.py) | 10,055 B | Automated packaging tool; generates `submission_manifest.csv`. | **KEEP_REQUIRED** | Synchronizes and verifies the standalone academic submission directory. |
| [`run_dashboard.bat`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/run_dashboard.bat) | 716 B | Windows 1-click batch launcher for `server.py`. | **KEEP_REQUIRED** | Allows evaluators to run the Flask dashboard without manual command typing. |
| [`run_streamlit.bat`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/run_streamlit.bat) | 661 B | Windows 1-click batch launcher for `app.py`. | **KEEP_REQUIRED** | Allows evaluators to launch the legacy Streamlit visualization app directly. |
| [`run_tests.bat`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/run_tests.bat) | 635 B | Windows 1-click batch launcher for `pytest tests/ -v`. | **KEEP_REQUIRED** | Automated test execution batch runner for project validation. |
| [`requirements.txt`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/requirements.txt) | 233 B | Verified dependency list (Flask, Torch, XGBoost, Scapy, etc.). | **KEEP_REQUIRED** | Required for environment replication; all packages verified active. |
| [`.env.example`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/.env.example) | 354 B | Environment variable template. | **KEEP_REQUIRED** | Required for server configuration documentation. |
| [`pytest.ini`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/pytest.ini) | 42 B | Pytest configuration. | **KEEP_REQUIRED** | Configures test discovery and execution paths. |
| [`.gitignore`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/.gitignore) | 695 B | Git ignore rules. | **KEEP_REQUIRED** | Prevents committing `.venv`, large datasets, models, and caches. |

---

### Category 4: `KEEP_FOR_RESEARCH` (Research Evidence, Figures, Reports, Models & Datasets)

Every file in this category is part of the research evidence chain, academic dissertation, viva presentation, or reproducible ML pipeline:

#### A. Master Prompt Specification
| File | Size (Bytes) | Purpose & Content | Decision |
| :--- | :---: | :--- | :---: |
| [`m.txt`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/m.txt) | 32,427 B | **MSc Master Research Prompt Specification:** Contains complete project identity, academic constraints, mathematical formulations, experimental protocol, and viva objectives. | **KEEP_FOR_RESEARCH** |

#### B. Academic Screenshots & Research Figures
Every image file was cross-referenced across all documentation, HTML, presentations, and code:

| Image File | Size (Bytes) | Referenced By | Decision | Reason |
| :--- | :---: | :--- | :---: | :--- |
| `01_dashboard.png` | 161,358 B | `final_completion_report.md`, `submission_verification.md`, `capture_screenshots.py`, `generate_pptx.py`, `sync_submission_package.py`, `submission_manifest.csv` | **KEEP_FOR_RESEARCH** | Submission evidence: Full Web Dashboard view. |
| `02_csv_upload.png` | 91,181 B | `final_completion_report.md`, `capture_screenshots.py`, `submission_manifest.csv` | **KEEP_FOR_RESEARCH** | Submission evidence: CSV Batch Preview interface. |
| `03_xgboost_predictions.png` | 122,455 B | `final_completion_report.md`, `capture_screenshots.py`, `submission_manifest.csv` | **KEEP_FOR_RESEARCH** | Submission evidence: XGBoost classification table. |
| `04_vae_anomaly_scores.png` | 129,696 B | `final_completion_report.md`, `capture_screenshots.py`, `submission_manifest.csv` | **KEEP_FOR_RESEARCH** | Submission evidence: VAE anomaly scoring view. |
| `05_model_comparison.png` | 148,154 B | `final_completion_report.md`, `submission_verification.md`, `capture_screenshots.py`, `generate_pptx.py`, `sync_submission_package.py`, `submission_manifest.csv` | **KEEP_FOR_RESEARCH** | Submission evidence: Dual-model comparison & metrics. |
| `06_validation_error.png` | 100,641 B | `final_completion_report.md`, `capture_screenshots.py`, `submission_manifest.csv` | **KEEP_FOR_RESEARCH** | Submission evidence: Validation error handling. |
| `fig1_dataset_distribution.png` | 106,567 B | `README.md`, `app.py`, `visualisation.py`, `run_experiments.py`, `submission_manifest.csv` | **KEEP_FOR_RESEARCH** | Research Figure 1: CICIDS2017 class balance chart. |
| `fig2_roc_pr_curves.png` | 233,564 B | `README.md`, `app.py`, `generate_pptx.py`, `visualisation.py`, `frontend/index.html`, `submission_manifest.csv` | **KEEP_FOR_RESEARCH** | Research Figure 2: ROC and PR comparison curves. |
| `fig3_confusion_matrices.png` | 124,255 B | `README.md`, `app.py`, `visualisation.py`, `frontend/index.html`, `submission_manifest.csv` | **KEEP_FOR_RESEARCH** | Research Figure 3: Dual confusion matrices. |
| `fig4_anomaly_score_distributions.png` | 251,866 B | `README.md`, `app.py`, `visualisation.py`, `submission_manifest.csv` | **KEEP_FOR_RESEARCH** | Research Figure 4: Benign vs Attack MSE histograms. |
| `fig5_per_family_recall.png` | 142,617 B | `README.md`, `app.py`, `visualisation.py`, `submission_manifest.csv` | **KEEP_FOR_RESEARCH** | Research Figure 5: Per-attack-family breakdown. |
| `fig6_threshold_sensitivity.png` | 214,159 B | `README.md`, `app.py`, `visualisation.py`, `submission_manifest.csv` | **KEEP_FOR_RESEARCH** | Research Figure 6: Threshold sensitivity curve. |
| `fig7_distribution_shift_degradation.png` | 153,960 B | `README.md`, `app.py`, `visualisation.py`, `submission_manifest.csv` | **KEEP_FOR_RESEARCH** | Research Figure 7: Zero-day generalization shift. |
| `fig8_model_disagreement.png` | 105,286 B | `README.md`, `app.py`, `visualisation.py`, `submission_manifest.csv` | **KEEP_FOR_RESEARCH** | Research Figure 8: Model complementarity analysis. |
| `logo.svg` | 3,853 B | `README.md`, `frontend/index.html`, `submission_manifest.csv` | **KEEP_REQUIRED** | Application branding logo. |

#### C. Machine Learning Models & Processors (`artifacts/models/`)
- [`artifacts/models/xgboost_model.json`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/artifacts/models/xgboost_model.json): Trained supervised classifier.
- [`artifacts/models/vae_checkpoint.pt`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/artifacts/models/vae_checkpoint.pt): Trained PyTorch Tabular VAE weights & calibrated threshold.
- [`artifacts/models/flow_preprocessor.joblib`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/artifacts/models/flow_preprocessor.joblib): Fitted 70-feature scaler, medians, and schema.

#### D. Research Reports & Audit Trails (`reports/`)
- Academic dissertation: [`reports/academic_report.md`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/reports/academic_report.md)
- Statistical validation: [`reports/statistical_validation.md`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/reports/statistical_validation.md)
- Record reconciliation: [`reports/record_count_reconciliation.md`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/reports/record_count_reconciliation.md)
- Experimental results: [`reports/experimental_results.md`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/reports/experimental_results.md)
- Methodology notes: [`reports/methodology_notes.md`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/reports/methodology_notes.md)
- Final completion report: [`reports/final_completion_report.md`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/reports/final_completion_report.md)
- Live monitoring audit (Phases 1–5): [`reports/live_validation_audit/`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/reports/live_validation_audit/)
- Stage C implementation & sensitivity: [`reports/live_validation_audit/stage_c/`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/reports/live_validation_audit/stage_c/)

#### E. Viva Presentation & Defense Material (`viva/`)
- [`viva/Cognisec_AI_Viva.pptx`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/viva/Cognisec_AI_Viva.pptx): Master 15-slide PowerPoint deck.
- [`viva/speaker_notes.md`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/viva/speaker_notes.md): Oral presentation notes.
- [`viva/viva_questions_and_answers.md`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/viva/viva_questions_and_answers.md): 25 academic defense questions & answers.

---

### Category 5: `REVIEW_REQUIRED`

None. All files have been definitively classified based on their imports, references, and operational role.

---

## 3. Summary of Proposed Changes (Stage B Review)

| Action | Target Items | Recoverable Space | Impact on Code / ML / Research |
| :--- | :--- | :---: | :--- |
| **Purge Build Cache** | Delete `build/` | 129,419 B (126.4 KB) | **Zero.** Ephemeral setuptools build directory. |
| **Purge Pytest Cache** | Delete `.pytest_cache/` | 2,527 B (2.5 KB) | **Zero.** Re-created automatically on next `pytest` run. |
| **Purge Bytecode Caches** | Delete root & `src/**/__pycache__/` & `tests/**/__pycache__/` | 552,774 B (539.8 KB) | **Zero.** Python recompiles bytecode on execution. |
| **Clean Pre-Fix Backup** | Delete `src/live/flow_extractor.py.bak_stage_c` | 22,283 B (21.8 KB) | **Zero.** Fixes are tested, committed, and pushed to GitHub remote (`22fa560`). |
| **Retain Virtualenv** | Preserve `.venv/` untouched | 0 B (Retained) | Ensures immediate out-of-the-box test and app execution. |
| **Retain All Images** | Preserve 6 screenshots, 8 research figures, 1 SVG logo | 0 B (Retained) | Preserves 100% of documentation, UI, presentation, and submission evidence. |
| **Retain All Scripts** | Preserve `app.py`, `main.py`, `server.py`, `*.bat`, etc. | 0 B (Retained) | Preserves all documented application entry points. |
| **Retain Master Prompt** | Preserve `m.txt` | 0 B (Retained) | Preserves primary research specification. |
| **TOTAL SPACE RECOVERED** | **5 Items / Directories** | **~707,003 B (~0.67 MB)** | **100% Safe, Non-Destructive, and Compliant.** |

---

## 4. Next Step: Awaiting User Approval

Per the **Stage B Review** requirement:
- Review the summary above and the proposed deletion of only the ephemeral caches (`build/`, `.pytest_cache/`, `__pycache__/`, and `flow_extractor.py.bak_stage_c`).
- Upon your approval, Stage C (Safe Deletion) and Stage D (Full Regression Verification) will be executed immediately.
