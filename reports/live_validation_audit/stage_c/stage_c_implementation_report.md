# Stage C Implementation Report: Controlled Feature-Semantics Fixes

**Project:** Cognisec AI: Network Behaviour Fingerprinting for Unseen Attack Detection  
**Phase:** Stage C — Controlled Code Implementation & Regression Validation  
**Date:** October 16, 2026  
**Auditor / Developer:** Santhoskrishna / Cognisec AI Team  
**Status:** **STAGE C IMPLEMENTATION & TESTING COMPLETE — AWAITING REVIEW**  

---

## 1. Executive Summary

In Stage C, controlled modifications were applied to [`src/live/flow_extractor.py`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/src/live/flow_extractor.py) to correct the upstream semantic discrepancies identified during the Stage A diagnostic audit. 

All corrections bring the live packet feature extractor into exact mathematical and architectural alignment with the authoritative Java implementation of **CICFlowMeter** ([`ahlashkari/CICFlowMeter`](https://github.com/ahlashkari/CICFlowMeter) `BasicFlow.java` and `BasicPacketInfo.java`), which generated the CICIDS2017 training benchmark.

### Strict Invariance Assertions
1. **Model Artifacts Untouched:** Neither [`artifacts/models/xgboost_model.json`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/artifacts/models/xgboost_model.json), [`artifacts/models/vae_checkpoint.pt`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/artifacts/models/vae_checkpoint.pt), nor [`artifacts/models/flow_preprocessor.joblib`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/artifacts/models/flow_preprocessor.joblib) were modified, retrained, or overwritten (`git status` confirms zero modifications to `artifacts/models/`).
2. **Thresholds Preserved:** Calibrated decision thresholds remain strictly unchanged at **`0.5200`** (XGBoost) and **`0.330610`** (PyTorch Tabular VAE).
3. **No Retraining or Fabrication:** No synthetic data was generated for academic metrics, and no experimental results were overwritten.
4. **Pre-Change Backup Created:** A timestamped backup was saved to [`src/live/flow_extractor.py.bak_stage_c`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/src/live/flow_extractor.py.bak_stage_c).

---

## 2. Confirmed Defects Fixed & Feature-Level Changes

| Finding ID | Severity | File & Section | Source Code Evidence | Description of Defect | Exact Change Implemented |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **FIND-01** | **CRITICAL** | `flow_extractor.py` &rarr; `to_features_dict()` | `BasicFlow.java:137, 151`<br>`fwdPktStats.addValue(packet.getPayloadBytes())` | Packet length statistics and byte totals used raw wire lengths (`p.wire_len`), giving 54–66 bytes to pure control packets (SYN, ACK, FIN, RST). | Changed `fwd_lens = [float(p.payload_len) for p in fwd]`, `bwd_lens = [float(p.payload_len) for p in bwd]`, and `all_lens = [float(p.payload_len) for p in all_pkts]`. Pure control packets now register 0 payload bytes. |
| **FIND-02** | **HIGH** | `flow_extractor.py` &rarr; `process_scapy_packet()` & `to_features_dict()` | `BasicFlow.java:148, 152`<br>`fHeaderBytes = packet.getHeaderBytes()` | Accumulated IP (20B) + Transport (20B) headers, reporting 40–52B instead of transport-layer header size (20B for TCP, 8B for UDP). | Stored `transport_header_len` in `PacketRecord.header_len`. Header features (`Fwd Header Length`, `Bwd Header Length`, `min_seg_size_forward`) now strictly evaluate to transport-layer header bytes (20 or 32 for TCP, 8 for UDP). |
| **FIND-03** | **MEDIUM** | `flow_extractor.py` &rarr; `to_features_dict()` | `BasicFlow.java:243`<br>`(double)(backward.size() / forward.size())` | Continuous floating-point division was used, yielding fractions (e.g. 0.625) whereas CICFlowMeter produces integer quotients (0.0). | Implemented `down_up_ratio = float(tot_bwd_pkts // tot_fwd_pkts) if tot_fwd_pkts > 0 else 0.0`. |
| **FIND-04** | **LOW** | `flow_extractor.py` &rarr; `to_features_dict()` | `BasicFlow.java:214-227`<br>`duration > 0L ? bytes / dur : 0.0` | Zero-duration single-packet flows used an artificial duration floor of $10^{-6}\text{ s}$, creating extreme rates ($1,000,000\text{ pkts/s}$). | Implemented safe zero-duration branch: when `duration_sec <= 0.0`, all four rate features evaluate cleanly to `0.0`. |

### Downstream Dependent Features Updated Concurrently
Updating packet length statistics to payload bytes automatically and consistently corrected all 21 dependent features:
1. `Total Length of Fwd Packets` (now sum of forward payload bytes)
2. `Total Length of Bwd Packets` (now sum of backward payload bytes)
3. `Fwd Packet Length Max`
4. `Fwd Packet Length Min` (now 0.0 for pure control packets)
5. `Fwd Packet Length Mean`
6. `Fwd Packet Length Std`
7. `Bwd Packet Length Max`
8. `Bwd Packet Length Min` (now 0.0 for pure control packets)
9. `Bwd Packet Length Mean`
10. `Bwd Packet Length Std`
11. `Min Packet Length` (now 0.0 for pure control packets)
12. `Max Packet Length`
13. `Packet Length Mean`
14. `Packet Length Std`
15. `Packet Length Variance`
16. `Flow Bytes/s` (payload bytes / duration)
17. `Average Packet Size` (total payload bytes / total packet count)
18. `Avg Fwd Segment Size` (fwd_mean)
19. `Avg Bwd Segment Size` (bwd_mean)
20. `Subflow Fwd Bytes` (matches Total Length of Fwd Packets)
21. `Subflow Bwd Bytes` (matches Total Length of Bwd Packets)

---

## 3. Automated Test Results

### Targeted Stage C Unit & Regression Tests
Added 6 deterministic unit and regression tests in [`tests/test_live_monitoring.py`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/tests/test_live_monitoring.py) verifying all 10 specified behaviors:
1. `test_unit_control_vs_payload_packet_lengths`: Verifies pure TCP control packets yield 0 payload length and packets with payload yield correct length statistics.
2. `test_unit_header_length_semantics_tcp_and_udp`: Verifies header lengths use transport layer header size (TCP=20, UDP=8) and do not include the 20-byte IPv4 header.
3. `test_unit_flow_rates_and_zero_duration`: Verifies single-packet and zero-duration flows produce rate=0.0 rather than arbitrary extreme numbers.
4. `test_unit_down_up_ratio_integer_division`: Verifies Down/Up Ratio matches Java integer division (8 fwd / 5 bwd &rarr; 0.0, 3 fwd / 4 bwd &rarr; 1.0, 3 fwd / 6 bwd &rarr; 2.0).
5. `test_unit_schema_ordering_and_inference_pipeline`: Verifies all 70 features are correctly accepted by `FlowPreprocessor` and produce valid XGBoost and VAE inference outputs without non-finite values.
6. `test_regression_csv_upload_inference_intact`: Confirms existing CSV upload endpoints remain fully functional.

### Full Test Suite Execution Output
Executed command:
`.\.venv\Scripts\python.exe -m pytest -q`
```
................................                                         [100%]
32 passed in 9.65s
```
- **Exit Code:** `0`
- **Total Tests:** 32
- **Passed:** 32 (100%)
- **Failed:** 0
- **Skipped:** 0
- **Execution Duration:** 9.65 seconds

### End-to-End Live HTTP Server Integration Verification
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
    [OK] Live flows query returned 1 flows.

>>> ALL LIVE HTTP SERVER ENDPOINTS AND INFERENCE PIPELINES VERIFIED 100% SUCCESSFUL! <<<
```

---

## 4. Remaining Unverified Feature Definitions & Academic Limitations

1. **Absence of Raw PCAP Files in Repository:**
   As established in Stage A, the repository contains raw CICIDS2017 CSV files and processed numpy arrays (`data/processed/*.npz`), but does not contain the original raw `.pcap` capture files (which exceed 50 GB). Direct byte-level PCAP replay against original UNB recordings is therefore unverified locally.
2. **Contemporary Protocol Distribution Shift (TLS 1.3 / HTTP/3 over QUIC UDP):**
   Modern web traffic exhibits structural packet length, handshake, and burst distributions that differ from the 2017 campus baseline. An autoencoder trained strictly on 2017 benign traffic will inherently produce higher reconstruction error on modern QUIC/TLS 1.3 flows than on HTTP/1.1 or TLS 1.2 traffic. This is a scientific property of generative density estimation and must not be masked by artificially relaxing decision thresholds.
3. **Unlabelled Live Traffic Status:**
   Live traffic captured on arbitrary interfaces lacks ground-truth attack labels. Any live flows flagged by XGBoost or VAE must continue to be labelled with the mandatory disclaimer:
   `"Model Prediction Only (No Ground-Truth Label)"`.

---

## 5. Instructions for Running a New Benign-Traffic Capture Session

Now that the feature extractor semantics have been corrected and verified, a fresh benign traffic validation capture can be conducted using the following non-destructive procedure:

### Option A: Using the Interactive Web UI
1. Ensure the Flask server is running:
   ```powershell
   .\.venv\Scripts\python.exe server.py --host 127.0.0.1 --port 5000
   ```
2. Open `http://127.0.0.1:5000` in a web browser and scroll to **Section 8: Live Network Capture & Dual Inference**.
3. Select an active authorized network interface (e.g., your primary Wi-Fi or Ethernet adapter).
4. Set **Capture Duration** to 30 or 60 seconds.
5. Click **Start Live Capture**. Perform ordinary background web activity (e.g. visiting academic or news websites).
6. When the capture finishes, inspect the updated flow table:
   - Verify that standard TCP handshakes show `Fwd Packet Length Min = 0` and `Fwd Header Length = 20` or `32`.
   - Observe the VAE anomaly score distribution and joint decisions.
7. Click **Export Live Logs** to save the session JSON/CSV for audit preservation.

### Option B: Using the Verification Script
Run the automated live verification script:
```powershell
.\.venv\Scripts\python.exe tests/verify_live_server.py
```
This performs a safe 3-second live loopback capture cycle and validates telemetry and flow retrieval.
