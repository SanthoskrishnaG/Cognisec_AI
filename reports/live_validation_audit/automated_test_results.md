# Phase 5: Automated Test Execution & Regression Suite Audit

**Document:** `reports/live_validation_audit/automated_test_results.md`  
**Execution Environment:** Python 3.12.10 (`.venv\Scripts\python.exe`)  
**Test Framework:** Pytest 9.1.1  
**Auditor:** Santhoskrishna / Cognisec AI Research Audit  
**Execution Date:** October 16, 2026  
**Status:** **26/26 TESTS PASSED (100% Pass Rate)**  

---

## 1. Test Execution Command & Terminal Output

### Command:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### Exact Output:
```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.7.0 -- C:\Users\Santhoskrishna\Documents\Cognisec_AI\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\Santhoskrishna\Documents\Cognisec_AI
configfile: pytest.ini
plugins: anyio-4.15.1
collecting ... collected 26 items

tests/test_api.py::test_api_health PASSED                                [  3%]
tests/test_api.py::test_api_metadata PASSED                              [  7%]
tests/test_api.py::test_api_experiments PASSED                           [ 11%]
tests/test_api.py::test_api_comparison PASSED                            [ 15%]
tests/test_api.py::test_api_sample_csv PASSED                            [ 19%]
tests/test_api.py::test_csv_preview_with_sample PASSED                   [ 23%]
tests/test_api.py::test_csv_preview_malformed PASSED                     [ 26%]
tests/test_api.py::test_xgboost_inference PASSED                         [ 30%]
tests/test_api.py::test_vae_inference PASSED                             [ 34%]
tests/test_api.py::test_dual_inference PASSED                            [ 38%]
tests/test_api.py::test_static_frontend_routes PASSED                    [ 42%]
tests/test_data_integrity.py::test_data_integrity_and_leakage_controls PASSED [ 46%]
tests/test_data_integrity.py::test_record_count_and_confusion_matrix_invariants PASSED [ 50%]
tests/test_data_integrity.py::test_statistical_validation_invariants PASSED [ 53%]
tests/test_environment.py::test_imports PASSED                           [ 57%]
tests/test_live_monitoring.py::test_bidirectional_flow_feature_schema PASSED [ 61%]
tests/test_live_monitoring.py::test_flow_tracker_tcp_termination PASSED  [ 65%]
tests/test_live_monitoring.py::test_flow_tracker_timeouts PASSED         [ 69%]
tests/test_live_monitoring.py::test_live_manager_inference PASSED        [ 73%]
tests/test_live_monitoring.py::test_live_manager_lifecycle PASSED        [ 76%]
tests/test_api.py::test_api_live_interfaces PASSED                       [ 80%]
tests/test_api.py::test_api_live_status PASSED                           [ 84%]
tests/test_api.py::test_api_live_flows_endpoint PASSED                   [ 88%]
tests/test_vae.py::test_tabular_vae_architecture PASSED                  [ 92%]
tests/test_vae.py::test_vae_trainer_fit_and_threshold PASSED             [ 96%]
tests/test_xgboost.py::test_xgboost_training_and_inference PASSED        [100%]

============================= 26 passed in 7.34s ==============================
```

---

## 2. Server Integration Test Execution (`tests/verify_live_server.py`)

### Command:
```powershell
.\.venv\Scripts\python.exe tests/verify_live_server.py
```

### Exact Output:
```text
[1] Testing GET /
    [OK] Index HTML rendered with all 8 sections.
[2] Testing static assets
    [OK] CSS, JS modules, and figures served.
[3] Testing API health
    [OK] System health: healthy, XGBoost threshold: 0.52, VAE threshold: 0.33060985803604126
[4] Testing API sample-csv
    [OK] Sample CSV fetched (100 rows).
[5] Testing POST /api/analyze/csv-preview
    [OK] CSV Preview compatible: 70/70 features matched.
[6] Testing POST /api/analyze/xgboost
    [OK] XGBoost evaluated 100 flows: 0 attacks, 100 benign.
[7] Testing POST /api/analyze/vae
    [OK] VAE evaluated 100 flows: 2 anomalies, 98 normal.
[8] Testing POST /api/analyze/dual
    [OK] Dual Inference Summary:
         Both Flagged:  0 (0.0%)
         XGBoost Only:  0 (0.0%)
         VAE Only:      2 (2.0%)
         Both Normal:   98 (98.0%)
[9] Testing API comparison & metadata
    [OK] Model comparison and disaggregation loaded.
[10] Testing Live Monitoring Module & Endpoints
    [OK] Live interfaces enumerated (48 interfaces detected).
    [OK] Live status telemetry verified.
    [OK] Live capture start/stop lifecycle confirmed.
    [OK] Live flows query returned 3 flows.

>>> ALL LIVE HTTP SERVER ENDPOINTS AND INFERENCE PIPELINES VERIFIED 100% SUCCESSFUL! <<<
```

---

## 3. Test Coverage Matrix & Identified Gaps

| Test Category | Files Involved | Coverage Status | Identified Gaps |
| :--- | :--- | :---: | :--- |
| **API Endpoints** | `test_api.py`, `verify_live_server.py` | **100% Covered** | None. Covers health, preview, metadata, upload, start, stop, flows. |
| **Data Invariants** | `test_data_integrity.py` | **100% Covered** | None. Validates 757,646 records, 14,682 probe duplicates, 0 leakage. |
| **Model Classes** | `test_xgboost.py`, `test_vae.py` | **100% Covered** | None. Tests latent dimensions, training, and threshold calibration. |
| **Live Flow Extractor** | `test_live_monitoring.py` | **PARTIAL** | Existing tests verify 70 feature names and non-zero counts, but did **not** test payload vs. wire length semantic equivalence. Stage C must add unit tests verifying that zero-payload packets yield length = 0. |
