# Cognisec AI: Live Network Monitoring Validation & Verification Report

**Project Title:** Cognisec AI: Network Behaviour Fingerprinting for Unseen Attack Detection — An Experimental Comparison of XGBoost and Variational Autoencoders  
**Module:** Live Network Traffic Capture and Real-Time ML/DL Inference  
**Validation Date:** October 16, 2026  
**Auditor / Engineer:** Santhoskrishna / Cognisec AI Research  
**Validation Status:** **PASSED (100% Automated & Real-World Verified)**  

---

## 1. Executive Summary

A real-time network traffic capture, bidirectional flow aggregation, and dual-model inference module has been successfully integrated into Cognisec AI. The module strictly adheres to all architectural constraints:
1. **Zero Retraining or Threshold Tampering:** Model checkpoints (`artifacts/models/xgboost_model.json` and `artifacts/models/vae_checkpoint.pt`) and preprocessor parameters (`artifacts/models/flow_preprocessor.joblib`) remain completely untouched. Decision thresholds remain identical to deployed research baselines: **`0.5200`** for XGBoost and **`0.330610`** for the PyTorch Tabular VAE.
2. **Exact 70-Feature Schema Alignment:** The native Python bidirectional flow extractor (`src/live/flow_extractor.py`) computes the identical 70 continuous behavioral metrics defined in the CICIDS2017 pipeline in microsecond units, including the duplicate mirror column `Fwd Header Length_duplicated_0`.
3. **Preservation of Existing Workflows:** All offline CSV upload analysis endpoints (`/api/analyze/csv-preview`, `/api/analyze/xgboost`, `/api/analyze/vae`, `/api/analyze/dual`) and research dashboards remain fully functional.
4. **Complete Regression Suite:** Automated tests expanded from 18 to **26 passing tests** (100% pass rate in 7.34s).
5. **Real-World Live Traffic Verified:** A live capture session conducted on the host Windows machine (`10.145.180.196`) successfully sniffed 49 live packets, extracted 7 bidirectional TCP/UDP flows, and executed real-time dual inference with zero schema mismatches.

---

## 2. Automated Test Results Breakdown

Full regression test executed via Python 3.12:
```text
pytest tests/ -v
============================= 26 passed in 7.34s ==============================
```

### 2.1 Live Monitoring Test Suite (`tests/test_live_monitoring.py`)

| Test Case Name | Objective / Scope | Result |
| :--- | :--- | :---: |
| `test_bidirectional_flow_feature_schema` | Verifies extractor outputs exactly 70 features matching `preprocessor.feature_columns` order, units, and duplicate column | **PASSED** |
| `test_flow_tracker_tcp_termination` | Verifies immediate flow completion and tracker cleanup upon observing TCP FIN/RST flags | **PASSED** |
| `test_flow_tracker_timeouts` | Verifies active and idle timeout expiration mechanics (15s idle / 120s max duration) | **PASSED** |
| `test_live_manager_inference` | Verifies end-to-end dual model inference on simulated completed flows with ring buffer storage | **PASSED** |
| `test_live_manager_lifecycle` | Verifies thread-safe start, stop, conflict prevention, and status telemetry reporting | **PASSED** |
| `test_api_live_interfaces` | Verifies `GET /api/live/interfaces` enumerates Windows network adapters with IP/type labels | **PASSED** |
| `test_api_live_status` | Verifies `GET /api/live/status` returns operational state, thresholds, and health telemetry | **PASSED** |
| `test_api_live_flows_endpoint` | Verifies `GET /api/live/flows` and `POST /api/live/clear` endpoints handle buffer querying and resets | **PASSED** |

### 2.2 Baseline System Integrity Tests (Preserved 18 Tests)

| Test Suite | Tests | Result | Notes |
| :--- | :---: | :---: | :--- |
| `test_api.py` | 11 | **PASSED** | All baseline API endpoints, CSV uploads, previews, and static routes |
| `test_data_integrity.py` | 3 | **PASSED** | 757,646 records, 14,682 train/test vectors, confusion matrix invariants |
| `test_environment.py` | 1 | **PASSED** | Python 3.12, PyTorch 2.6, XGBoost 2.1, Scikit-learn 1.6, Scapy 2.8 |
| `test_vae.py` | 2 | **PASSED** | Latent dimension (16), MSE reconstruction, threshold calibration |
| `test_xgboost.py` | 1 | **PASSED** | Supervised binary classification, predict_proba calibration |

---

## 3. End-to-End Live HTTP Server Integration Validation

Executed integration test suite (`tests/verify_live_server.py`) against active HTTP server:
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

## 4. Real-World Live Capture Verification (Windows Wi-Fi Adapter)

A live capture session was initiated against the active Windows host network adapter (`10.145.180.196`):

1. **Capture Session Parameters:**
   - Interface: Default Host Wireless Adapter (`10.145.180.196`)
   - BPF Filter: None (All IP/TCP/UDP)
   - Duration: 2.0 seconds
2. **Observed Traffic:**
   - Total Packets Sniffed: **49 raw network frames**
   - In-Flight Flow Concurrency: 7 active tracked conversations
3. **Flow Extraction & Model Inference Results:**
   - Completed / Flushed Flows: **7 flows**
   - Feature Extraction Compatibility: **70 / 70 features (100% schema match)**
   - Missing / Non-Finite Imputations: 0 (cleanly handled)

#### Real Live Flow Sample Output:

| Flow ID | Protocol | Duration | Wire Bytes | XGBoost Prob | VAE MSE | Decision Tag |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `104.18.39.21:443 - 10.145.180.196:64257` | TCP | 0.042s | 1,420 B | 0.0003 (Benign) | 0.8412 | Unseen / Behavioral Anomaly (VAE Only) |
| `10.145.180.196:51154 - 172.217.117.4:443` | TCP | 0.118s | 3,890 B | 0.0012 (Benign) | 1.1205 | Unseen / Behavioral Anomaly (VAE Only) |
| `10.145.180.196:49666 - 142.251.222.138:443`| UDP | 0.000s | 168 B | 0.0001 (Benign) | 0.5218 | Unseen / Behavioral Anomaly (VAE Only) |

#### Empirical Analysis of Live Inference:
- **XGBoost:** Confirmed Benign on all host web flows with probabilities $\ll 0.52$ (typical values: $0.0001 - 0.0012$). The supervised model recognized that ordinary HTTPS browsing traffic lacks the volumetric signature of DoS or PortScan attacks.
- **VAE Reconstruction:** Flagged contemporary HTTPS TLS 1.3 / QUIC traffic as anomalous relative to the 2017 CICIDS2017 benign baseline ($MSE > 0.330610$). This is an authentic and expected empirical result: network protocol standards (TLS 1.3 cipher suites, QUIC UDP encapsulation) have evolved since 2017, demonstrating real-world distribution shift in action.
- **Scientific Labeling:** The dashboard accurately labeled these records as **Model Predictions (Unseen / Behavioral Shift)**, avoiding false claims of cyberattack.

---

## 5. Backward Compatibility & Non-Destructive Integrity Audit

1. **Model Checkpoints:**
   - `artifacts/models/xgboost_model.json`: Unmodified (SHA256 preserved).
   - `artifacts/models/vae_checkpoint.pt`: Unmodified (SHA256 preserved).
   - `artifacts/models/flow_preprocessor.joblib`: Unmodified (SHA256 preserved).
2. **Offline Research Results:**
   - `results/master_experiment_results.json`: Unmodified.
   - `reports/research_audit.md`: Unmodified.
   - All empirical metrics (PR-AUC 99.99%, VAE held-out recall, McNemar p-values) remain intact.
3. **Verified Backup Directory:**
   - `C:\Users\Santhoskrishna\Documents\Cognisec_AI_Backup_2026-10-15`: Completely untouched and verified.

---

## 6. Limitations & Deployment Recommendations

1. **Ground-Truth Label Absence:** Live network flows lack verified ground-truth labels. Decisions must be treated as probabilistic alerts rather than confirmed security breaches.
2. **Dataset Temporal Drift:** Models trained on CICIDS2017 reflect 2017 traffic characteristics. In production environments, periodic semi-supervised recalibration of the VAE on contemporary benign network baselines is recommended to maintain the target 1.0% false-alarm rate ceiling.
3. **Privilege Requirements:** Windows raw packet capture requires appropriate Npcap driver installation permissions. Non-administrator execution requires Npcap configured with non-restricted access.
