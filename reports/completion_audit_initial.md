# Cognisec AI — Baseline Completion Audit (Initial)

**Project:** Cognisec AI: Network Behaviour Fingerprinting for Unseen Attack Detection — An Experimental Comparison of XGBoost and Variational Autoencoders  
**Audit Date:** 2026-10-16  
**Auditor:** Senior Research Auditor & ML Systems Engineer  
**Environment:** Windows 10/11 | Python 3.12.10 (`.venv\Scripts\python.exe`)  
**Repository Branch:** `main` (commit `b0e5a5e`)

---

## 1. Executive Summary of Project State

| Component | Status | Empirical Evidence / Health Check |
| :--- | :--- | :--- |
| **Python Environment** | **HEALTHY** | Isolated `.venv` (Python 3.12.10) with `torch 2.14.1+cpu`, `xgboost 3.4.1`, `scikit-learn 1.9.1`, `pandas 3.0.6`, `numpy 2.2.6`, `flask 3.1.3`. Auto-resolution active in `main.py`, `server.py`, and batch scripts. |
| **Model Artifacts** | **VERIFIED** | `flow_preprocessor.joblib` (70 features), `xgboost_model.json` (threshold 0.52), `vae_checkpoint.pt` (threshold 0.330610). Verified load and inference on authentic flows. |
| **Dataset Partitions** | **VERIFIED** | 375,262 Train / 125,087 Val / 257,297 Test records stored in `data/processed/*.npz` with `partition_manifest.json`. Zero held-out attack leakage in Train/Val splits. |
| **Automated Tests** | **PASSING** | 16/16 unit and integration tests passed in 7.35s (`pytest tests/ -v`). |
| **Flask API & UI** | **OPERATIONAL** | Flask server listening on `http://127.0.0.1:5000` with 8 SPA dashboard views, live CSV preview, SVG charts, and XGBoost/VAE dual inference. |
| **Audit Focus Areas** | **IN PROGRESS** | 1) Record-count discrepancy (125,088 known + 172,209 held-out = 297,297 vs 257,297 test size); 2) Rigorous statistical validation of paired comparisons; 3) Academic documentation & viva package completion. |

---

## 2. File & Artifact Baseline Inventory (with SHA-256 Hashes)

### Model Artifacts (`artifacts/models/`)
- `flow_preprocessor.joblib` (4,838 bytes)  
  `SHA-256: a9bc7f32ce1806a10081aa1d755a7872d8df528a32d04d89fdc42b24d82e22cf`
- `vae_checkpoint.pt` (74,453 bytes)  
  `SHA-256: 3e824e46a29b739eeee3dea8ad08fff24c27534bf53def02ed0de7cf8820c0e7`
- `xgboost_model.json` (638,017 bytes)  
  `SHA-256: 5f6e61c8dde5aaffd88fd4471eba744a94307fb58e5d103fbed887ca1264fe52`
- `xgboost_model.meta.json` (2,147 bytes)  
  `SHA-256: 4ab4fa021b49b17ec9b107365455b777f6b6267133a84eda7ec7fcd7b4f390c8`

### Processed Partitions (`data/processed/`)
- `partition_manifest.json` (1,251 bytes)  
  `SHA-256: 7d33d30d72b51a03799139f3fa58d9a49c1abbea48bc5a95c182b6643f915fde`
- `train_data.npz` (54,846,232 bytes, 375,262 samples)  
  `SHA-256: 413261921f60f56d577b10a82e0f50fedcc6a6c40cacf76d1979809f13435dd9`
- `val_data.npz` (18,374,129 bytes, 125,087 samples)  
  `SHA-256: acf3821f0f41afdf974395ff7c53ef410ea92a3278453a46767ab319c216bb63`
- `test_data.npz` (38,594,631 bytes, 257,297 samples)  
  `SHA-256: 1df3ac5a8fd103a01db186842d2df491bb3b85920306d0cc927e9e319f4f8128`

### Raw CICIDS2017 Datasets (`data/raw/`)
- `Friday-WorkingHours-Afternoon-DDos.csv` (57,055,277 bytes) | `SHA-256: 975dfab7dd6447977c2b9474afc872cfcb6dc83a87603393290c5046b79995e2`
- `Friday-WorkingHours-Afternoon-PortScan.csv` (52,205,654 bytes) | `SHA-256: 08dbc58a8551e8d94826c7a9acf328828aa5046c5cb6afc702fa56c97fcdfb21`
- `Friday-WorkingHours-Morning.csv` (9,014,478 bytes) | `SHA-256: 97861740269db3e02d431c7dd3d29269cd1212c84c79b5280ade0668c2acf15a`
- `Monday-WorkingHours.csv` (9,029,716 bytes) | `SHA-256: 3f9f2ad217dbbc6fa3bb25070312c6d20512da48af30d9003dbe4d58dc017862`
- `Thursday-WorkingHours-Afternoon-Infilteration.csv` (7,979,975 bytes) | `SHA-256: 63ae73dc781daa2f7102f1d3517ea7b5d1b60c420e6a435d6df75df0aa9bd857`
- `Thursday-WorkingHours-Morning-WebAttacks.csv` (9,091,484 bytes) | `SHA-256: ab94606ff45104286a67855f721250fde2a8bfd1cd1de8b3ad9efd512bdde98d`
- `Tuesday-WorkingHours.csv` (13,229,339 bytes) | `SHA-256: 6644b8bf2a5d713d37948350959f2042f055fc90f53e183da2dab90e8fc35867`
- `Wednesday-workingHours.csv` (105,302,558 bytes) | `SHA-256: c9eb37024ff893abfa4d5aaeeb7b261a7f1a8c22dbf1cdfb251ea637b470382f`

### Primary Experiment Results (`results/`)
- `master_experiment_results.json` (8,555 bytes) | `SHA-256: 94e5e90e486123518cbc46dbbae03dd4eaa92455e77f9d46f14f6894d740ec06`
- `reproducibility_audit_report.json` (5,443 bytes) | `SHA-256: 837d9fac231a6a43b59db04f2da20f10ba7a65e0a73e5ba7b2e5f706449a7582`

---

## 3. Baseline Test Suite Verification

Command: `.\.venv\Scripts\python.exe -m pytest tests/ -v`
```
tests/test_api.py::test_api_health PASSED                                [  6%]
tests/test_api.py::test_api_metadata PASSED                              [ 12%]
tests/test_api.py::test_api_experiments PASSED                           [ 18%]
tests/test_api.py::test_api_comparison PASSED                            [ 25%]
tests/test_api.py::test_api_sample_csv PASSED                            [ 31%]
tests/test_csv_preview_with_sample PASSED                                [ 37%]
tests/test_csv_preview_malformed PASSED                                  [ 43%]
tests/test_xgboost_inference PASSED                                      [ 50%]
tests/test_vae.py::test_vae_inference PASSED                             [ 56%]
tests/test_api.py::test_dual_inference PASSED                            [ 62%]
tests/test_api.py::test_static_frontend_routes PASSED                    [ 68%]
tests/test_data_integrity.py::test_data_integrity_and_leakage_controls PASSED [ 75%]
tests/test_environment.py::test_imports PASSED                           [ 81%]
tests/test_vae.py::test_tabular_vae_architecture PASSED                  [ 87%]
tests/test_vae.py::test_vae_trainer_fit_and_threshold PASSED             [ 93%]
tests/test_xgboost.py::test_xgboost_training_and_inference PASSED        [100%]
============================= 16 passed in 7.35s ==============================
```

---

## 4. Work Plan & Execution Order
1. **Task 2:** Reconcile record-count discrepancy between top-level 257,297 and sum of evaluations (125,088 + 172,209 = 297,297).
2. **Task 3:** Audit and calculate statistically rigorous paired comparisons (McNemar's test & paired bootstrap CI).
3. **Tasks 4 & 5:** Write academic methodology, comprehensive experimental results, and generate figures.
4. **Task 6:** Formalize dataset citations and limitations.
5. **Tasks 7 & 8:** Assemble clean submission package and capture real browser screenshots.
6. **Task 9:** Expand regression testing suite.
7. **Tasks 10 & 11:** Generate viva PowerPoint presentation (`.pptx`), speaker notes, and 35+ defense Q&A.
8. **Task 12:** Complete final quality audit report.
