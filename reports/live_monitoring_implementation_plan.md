# Cognisec AI: Live Network Monitoring and Inference Implementation Plan

**Project Title:** Cognisec AI: Network Behaviour Fingerprinting for Unseen Attack Detection — An Experimental Comparison of XGBoost and Variational Autoencoders  
**Module:** Live Network Traffic Capture, Bidirectional Flow Extraction, and Real-Time ML/DL Inference  
**Author:** Santhoskrishna / Cognisec AI Engineering  
**Date:** October 2026  
**Status:** Architecture Plan & Audit Baseline  

---

## 1. Executive Summary & Audit Baseline

This document specifies the technical design for introducing real-time network flow monitoring and inference to Cognisec AI without retraining models, changing calibrated decision thresholds, or compromising existing CSV batch analysis capabilities.

### 1.1 Mandatory Initial Audit Findings

1. **Model Pipeline Inspection:**
   - **Supervised Model:** Extreme Gradient Boosting (`XGBClassifier`) loaded from `artifacts/models/xgboost_model.json`.
     - Input Representation: **Raw unscaled imputed features (`X_unscaled`)** of dimension $(N, 70)$, type `float32`.
     - Decision Function: Logistic sigmoid probability $\sigma(\hat{y}) \in [0.0, 1.0]$.
     - Calibrated Decision Threshold: **`0.52`** (established via validation grid search maximizing $F_1$-score).
     - Output: Attack Probability and binary classification (`Attack` if prob $\ge 0.52$ else `Benign`).
   - **Unsupervised Anomaly Model:** PyTorch Tabular Variational Autoencoder (`TabularVAE`) loaded from `artifacts/models/vae_checkpoint.pt`.
     - Input Representation: **Log1p + Robust/Standard scaled features (`X_scaled`)**, clipped to $[-5.0, 5.0]$, of dimension $(N, 70)$, type `float32`.
     - Decision Function: Per-flow Mean Squared Reconstruction Error ($MSE$).
     - Calibrated Decision Threshold: **`0.330610`** (calibrated at the 99.0th percentile of benign validation flows, enforcing an operational 1.0% false-alarm ceiling).
     - Output: Anomaly Score and decision (`Anomaly` if score $\ge 0.330610$ else `Normal`).
   - **Feature Preprocessor:** `FlowPreprocessor` loaded from `artifacts/models/flow_preprocessor.joblib`.
     - Exact Feature Count: Exactly **70 continuous behavioral statistical features**.
     - Imputation Policy: Training-set median values per feature; non-finite infinities replaced with median values.
     - Scaling Transformation: $X_{\text{log}} = \log(1 + \max(0, X)) \to \text{StandardScaler.transform}(X_{\text{log}})$.
   - **Baseline Test Suite:** Current test suite passes **18/18 tests** cleanly in 6.49s.

---

## 2. The Exact 70-Feature Schema & Order Specification

Both the XGBoost discriminator and the Tabular VAE encoder require the identical 70 features in the exact sequential order defined in `flow_preprocessor.joblib`:

| Index | Feature Column Name | Unit / Type | Description / Extraction Metric |
| :---: | :--- | :--- | :--- |
| **1** | `Destination Port` | Integer | TCP/UDP destination service port (forward direction) |
| **2** | `Flow Duration` | Microseconds | Total flow duration: $(t_{\text{last}} - t_{\text{first}}) \times 10^6$ |
| **3** | `Total Fwd Packets` | Integer | Total packets transmitted in forward direction (A $\to$ B) |
| **4** | `Total Backward Packets` | Integer | Total packets transmitted in backward direction (B $\to$ A) |
| **5** | `Total Length of Fwd Packets` | Bytes | Total payload/packet bytes in forward direction |
| **6** | `Total Length of Bwd Packets` | Bytes | Total payload/packet bytes in backward direction |
| **7** | `Fwd Packet Length Max` | Bytes | Maximum packet length in forward direction |
| **8** | `Fwd Packet Length Min` | Bytes | Minimum packet length in forward direction |
| **9** | `Fwd Packet Length Mean` | Bytes | Mean packet length in forward direction |
| **10** | `Fwd Packet Length Std` | Bytes | Standard deviation of forward packet lengths |
| **11** | `Bwd Packet Length Max` | Bytes | Maximum packet length in backward direction |
| **12** | `Bwd Packet Length Min` | Bytes | Minimum packet length in backward direction |
| **13** | `Bwd Packet Length Mean` | Bytes | Mean packet length in backward direction |
| **14** | `Bwd Packet Length Std` | Bytes | Standard deviation of backward packet lengths |
| **15** | `Flow Bytes/s` | Bytes/sec | Total flow bytes divided by duration in seconds |
| **16** | `Flow Packets/s` | Packets/sec | Total flow packets divided by duration in seconds |
| **17** | `Flow IAT Mean` | Microseconds | Mean inter-arrival time between all consecutive packets |
| **18** | `Flow IAT Std` | Microseconds | Standard deviation of flow inter-arrival times |
| **19** | `Flow IAT Max` | Microseconds | Maximum flow inter-arrival time |
| **20** | `Flow IAT Min` | Microseconds | Minimum flow inter-arrival time |
| **21** | `Fwd IAT Total` | Microseconds | Sum of inter-arrival times in forward direction |
| **22** | `Fwd IAT Mean` | Microseconds | Mean inter-arrival time in forward direction |
| **23** | `Fwd IAT Std` | Microseconds | Standard deviation of forward inter-arrival times |
| **24** | `Fwd IAT Max` | Microseconds | Maximum forward inter-arrival time |
| **25** | `Fwd IAT Min` | Microseconds | Minimum forward inter-arrival time |
| **26** | `Bwd IAT Total` | Microseconds | Sum of inter-arrival times in backward direction |
| **27** | `Bwd IAT Mean` | Microseconds | Mean inter-arrival time in backward direction |
| **28** | `Bwd IAT Std` | Microseconds | Standard deviation of backward inter-arrival times |
| **29** | `Bwd IAT Max` | Microseconds | Maximum backward inter-arrival time |
| **30** | `Bwd IAT Min` | Microseconds | Minimum backward inter-arrival time |
| **31** | `Fwd PSH Flags` | Count | Number of forward packets with TCP PSH flag set |
| **32** | `Fwd URG Flags` | Count | Number of forward packets with TCP URG flag set |
| **33** | `Fwd Header Length` | Bytes | Total TCP/IP header length in forward direction |
| **34** | `Bwd Header Length` | Bytes | Total TCP/IP header length in backward direction |
| **35** | `Fwd Packets/s` | Packets/sec | Forward packets divided by flow duration |
| **36** | `Bwd Packets/s` | Packets/sec | Backward packets divided by flow duration |
| **37** | `Min Packet Length` | Bytes | Minimum packet length across all flow packets |
| **38** | `Max Packet Length` | Bytes | Maximum packet length across all flow packets |
| **39** | `Packet Length Mean` | Bytes | Mean packet length across all flow packets |
| **40** | `Packet Length Std` | Bytes | Standard deviation of packet lengths |
| **41** | `Packet Length Variance` | $\text{Bytes}^2$ | Variance of packet lengths |
| **42** | `FIN Flag Count` | Count | Total packets with TCP FIN flag set |
| **43** | `SYN Flag Count` | Count | Total packets with TCP SYN flag set |
| **44** | `RST Flag Count` | Count | Total packets with TCP RST flag set |
| **45** | `PSH Flag Count` | Count | Total packets with TCP PSH flag set |
| **46** | `ACK Flag Count` | Count | Total packets with TCP ACK flag set |
| **47** | `URG Flag Count` | Count | Total packets with TCP URG flag set |
| **48** | `CWE Flag Count` | Count | Total packets with TCP CWE flag set |
| **49** | `ECE Flag Count` | Count | Total packets with TCP ECE flag set |
| **50** | `Down/Up Ratio` | Ratio | Backward packets divided by forward packets |
| **51** | `Average Packet Size` | Bytes | Average packet size across all flow packets |
| **52** | `Avg Fwd Segment Size` | Bytes | Average forward segment/packet payload size |
| **53** | `Avg Bwd Segment Size` | Bytes | Average backward segment/packet payload size |
| **54** | `Fwd Header Length_duplicated_0` | Bytes | Exact mirror copy of feature 33 (`Fwd Header Length`) |
| **55** | `Subflow Fwd Packets` | Integer | Forward subflow packet count (identical to feature 3) |
| **56** | `Subflow Fwd Bytes` | Bytes | Forward subflow byte count (identical to feature 5) |
| **57** | `Subflow Bwd Packets` | Integer | Backward subflow packet count (identical to feature 4) |
| **58** | `Subflow Bwd Bytes` | Bytes | Backward subflow byte count (identical to feature 6) |
| **59** | `Init_Win_bytes_forward` | Bytes | Initial TCP window advertisement in forward direction |
| **60** | `Init_Win_bytes_backward` | Bytes | Initial TCP window advertisement in backward direction |
| **61** | `act_data_pkt_fwd` | Count | Forward packets containing non-empty payload ($> 0$ bytes) |
| **62** | `min_seg_size_forward` | Bytes | Minimum TCP header length observed in forward direction |
| **63** | `Active Mean` | Microseconds | Mean duration flow was actively sending before idle |
| **64** | `Active Std` | Microseconds | Standard deviation of active periods |
| **65** | `Active Max` | Microseconds | Maximum active period |
| **66** | `Active Min` | Microseconds | Minimum active period |
| **67** | `Idle Mean` | Microseconds | Mean duration flow was idle between active bursts |
| **68** | `Idle Std` | Microseconds | Standard deviation of idle periods |
| **69** | `Idle Max` | Microseconds | Maximum idle period |
| **70** | `Idle Min` | Microseconds | Minimum idle period |

---

## 3. Live Capture & Flow Extraction Architecture

```text
       [ Network Interface ] (Wi-Fi, Ethernet, or Loopback)
               │
               ▼
       [ Scapy Packet Sniffer ] (Background Daemon Thread, Npcap backend)
               │  Packet Stream (TCP/UDP IP packets; NO payload saved)
               ▼
   [ Bidirectional Flow Extractor ] (src/live/flow_extractor.py)
         - 5-Tuple Key: (Src IP, Src Port, Dst IP, Dst Port, Proto)
         - Bi-directional association (forward vs. backward)
         - Flow Inactivity Timeout: 15 seconds
         - Flow Maximum Duration: 120 seconds
         - Immediate TCP FIN/RST termination
               │  Completed Bounded Flows
               ▼
     [ Live Feature Adapter ] (src/live/feature_adapter.py)
         - Computes 70 statistical behavioral features
         - Validates finite numbers, sets correct units (microseconds)
         - Feeds into fitted flow_preprocessor.joblib (zero refitting)
               │
               ├──> X_unscaled ──> [ Supervised XGBoost ] ──> Probability & Decision
               └──> X_scaled   ──> [ Tabular PyTorch VAE ] ─> Anomaly Score & Decision
                                              │
                                              ▼
                             [ Live Monitoring Manager ] (Ring Buffer)
                                              │
                                              ▼
                                   [ Flask REST API ]
                                 /api/live/start
                                 /api/live/stop
                                 /api/live/status
                                 /api/live/flows
                                 /api/live/interfaces
                                              │
                                              ▼
                         [ Frontend Live Monitoring View ]
```

---

## 4. Key Design Principles & Safeguards

1. **Pure-Python Scapy + Npcap Engine:**
   - Evaluated external collectors (e.g., standalone Java CICFlowMeter vs. native Scapy).
   - Java CICFlowMeter requires a JRE, writes temporary CSVs to disk, and suffers known compatibility issues with newer Windows Npcap drivers.
   - A pure-Python bidirectional flow extractor using Scapy operates natively in `.venv`, requires no external runtime, processes packets in-memory, and guarantees exact feature-name and unit alignment.
2. **Authorized Local Capture Only:**
   - The user selects an explicit network interface (e.g., active Wi-Fi `10.145.180.196` or Loopback `127.0.0.1`).
   - The system captures traffic passively; it never sends probes, performs port scans, or injects packets into external networks.
3. **No Packet Payload Retention:**
   - In accordance with cybersecurity ethics and data protection, packet payload contents (application data, passwords, HTTP bodies) are immediately discarded after computing header lengths and payload sizes.
4. **Completed Flows vs. Raw Packets:**
   - The system aggregates packets into completed flows. An individual packet is never passed to the ML models as if it were a flow.
5. **Thread-Safe Session Lifecycle:**
   - Capture runs in a dedicated background daemon thread.
   - A global session lock ensures only one capture session can be active at any given time.
   - Ring buffers with bounded capacities (e.g., 2,000 recent flows) prevent unbounded memory consumption.
6. **Ground-Truth Transparency:**
   - Dashboard outputs are explicitly labeled as **Model Predictions**, not confirmed cyberattacks, acknowledging the absence of ground-truth labels on live networks.

---

## 5. Modular Component Plan

| Component File | Directory | Purpose |
| :--- | :--- | :--- |
| `flow_extractor.py` | `src/live/` | Bidirectional flow tracker, timeout manager, and 70-feature statistical calculator |
| `live_manager.py` | `src/live/` | Thread-safe capture coordinator, ring buffer, interface enumeration, and ML runner |
| `live_routes.py` | `src/live/` | Flask Blueprint exposing `/api/live/*` endpoints |
| `live_monitoring.js` | `frontend/js/` | Frontend controller managing polling, start/stop buttons, counters, and real-time alerts table |
| `index.html` (View) | `frontend/` | New "Live Network Monitoring" navigation item and view section |
| `test_live_monitoring.py` | `tests/` | Mocked unit tests for flow assembly, feature adapter, inference, and Flask routes |

---

## 6. Implementation Steps

1. **Step 1:** Create `src/live/flow_extractor.py` implementing bidirectional packet aggregation and 70-feature statistical calculation.
2. **Step 2:** Create `src/live/live_manager.py` managing Scapy sniffing, thread lifecycle, interface discovery, and dual model inference.
3. **Step 3:** Create `src/live/live_routes.py` with Flask REST endpoints (`/api/live/interfaces`, `/api/live/start`, `/api/live/stop`, `/api/live/status`, `/api/live/flows`).
4. **Step 4:** Register live monitoring Blueprint in `src/api/server.py` while ensuring existing `/api/analyze/*` CSV upload routes remain completely unchanged.
5. **Step 5:** Add the Live Monitoring view to `frontend/index.html` and implement `frontend/js/live_monitoring.js`.
6. **Step 6:** Write unit tests in `tests/test_live_monitoring.py` using synthetic mock packets.
7. **Step 7:** Run full pytest regression suite (verifying both old 18 tests and new live tests pass).
8. **Step 8:** Execute a live verification capture on the local loopback/Wi-Fi interface and generate documentation.
