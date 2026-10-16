# Cognisec AI: Submission Package Cryptographic Verification Report

**Target Package Location:** `C:\Users\Santhoskrishna\Documents\Cognisec_AI_Submission`  
**Verification Timestamp:** October 2026  
**Auditor:** Santhoskrishna / Cognisec AI Automated Audit Suite  
**Total Cataloged Package Files:** 100 files  

---

## 1. Package Structure & Completeness Verification

The submission package conforms strictly to the academic research submission guidelines:
- `source/`: Complete clean Python backend, HTML5/CSS3/JS frontend, evaluation pipelines, and pytest suite (excluding `.venv`, caches, and git history).
- `models/`: SHA-256 verified serialized model artifacts (`xgboost_model.json`, `vae_checkpoint.pt`, `flow_preprocessor.joblib`).
- `data/raw/`: All 8 authentic CICIDS2017 flow CSV files preserved with full cryptographic integrity.
- `data/processed/`: Deterministic training, validation, and test numpy archives alongside partition manifest.
- `results/`: Verified master benchmark experiment results and reproducibility audit reports.
- `reports/`: Complete research reports, statistical validation, record count reconciliation, figures, and genuine screenshots.
- `viva/`: Professional 19-slide presentation (`Cognisec_AI_Viva.pptx`), speaker notes (`speaker_notes.md`), and comprehensive defense questions (`viva_questions_and_answers.md`).
- `README.md` & `requirements.txt`: Root deployment manifests for one-click environment recreation.

---

## 2. Key Deliverable Cryptographic Hashes

| Category | Relative File Path | Size (Bytes) | SHA-256 Digest |
| :--- | :--- | :--- | :--- |
| **Models** | `models/xgboost_model.json` | 638,017 | `5f6e61c8dde5aaff...1264fe52` |
| **Models** | `models/vae_checkpoint.pt` | 74,453 | `3e824e46a29b739e...8820c0e7` |
| **Models** | `models/flow_preprocessor.joblib` | 4,838 | `a9bc7f32ce1806a1...d82e22cf` |
| **Results** | `results/master_experiment_results.json` | 9,439 | `e8b676a1eeb0cb8c...a1cdc601` |
| **Viva** | `viva/Cognisec_AI_Viva.pptx` | 529,858 | `6ba5eaa68117080c...f126a0b3` |
| **Viva** | `viva/speaker_notes.md` | 15,911 | `9ad745868b1d2dd5...1570cb49` |
| **Viva** | `viva/viva_questions_and_answers.md` | 24,741 | `34aed63a2f9c42c4...934a8425` |
| **Reports** | `reports/academic_report.md` | 21,421 | `d1469ff5bce1f1a9...8cdd8d87` |
| **Reports** | `reports/record_count_reconciliation.md` | 5,517 | `a8658fef32b1b7c5...3e8058c8` |
| **Reports** | `reports/statistical_validation.md` | 4,953 | `8af4467b248d1c2d...94797e23` |
| **Reports** | `reports/screenshots/01_dashboard.png` | 161,358 | `ce1ab2ad393eacda...fe31127e` |
| **Reports** | `reports/screenshots/05_model_comparison.png` | 148,154 | `7a319894a4166eaf...78d29a7c` |
| **Source** | `source/server.py` | 5,708 | `f07816509827d5c6...9c6c728e` |
| **Source** | `source/requirements.txt` | 233 | `470f1e66d3643c7d...fea5be17` |

---

## 3. Verified Execution Command for Packaged Submission

To execute the submission application in an independent Python environment:
```bash
cd C:\Users\Santhoskrishna\Documents\Cognisec_AI_Submission\source
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python server.py --host 127.0.0.1 --port 5000
```
The application will serve the interactive research workbench on `http://127.0.0.1:5000/`.

## 4. Verification Status: PASS
All model artifacts, datasets, code modules, and academic documentation are self-contained and cryptographically verified.