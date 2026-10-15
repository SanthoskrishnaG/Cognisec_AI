# Stage C Fixture-Level Feature Difference & Model Sensitivity Report

**Document:** `reports/live_validation_audit/stage_c/fixture_level_differences.md`  
**Phase:** Stage C Verification  
**Evaluation Script:** [`reports/live_validation_audit/stage_c/run_fixture_comparison.py`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/reports/live_validation_audit/stage_c/run_fixture_comparison.py)  
**Data Artifact:** [`reports/live_validation_audit/stage_c/fixture_comparison_results.json`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/reports/live_validation_audit/stage_c/fixture_comparison_results.json)  

---

## 1. Overview

To verify the numerical impact of the Stage C corrections, both the baseline flow extractor ([`src/live/flow_extractor.py.bak_stage_c`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/src/live/flow_extractor.py.bak_stage_c)) and the corrected extractor ([`src/live/flow_extractor.py`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/src/live/flow_extractor.py)) were executed on four identical, deterministic network packet fixtures.

The extracted feature vectors were evaluated through the fitted [`FlowPreprocessor`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/artifacts/models/flow_preprocessor.joblib), [`XGBoostClassifierWrapper`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/artifacts/models/xgboost_model.json), and [`TabularVAE`](file:///C:/Users/Santhoskrishna/Documents/Cognisec_AI/artifacts/models/vae_checkpoint.pt).

---

## 2. Per-Fixture Quantitative Comparison

### Fixture 1: Clean 3-Packet TCP Handshake (0 Payload Bytes)
- **Packets:** SYN &rarr; SYN-ACK &larr; ACK &rarr;
- **Total Payload Bytes:** 0
- **Flow Duration:** 30 ms (30,000 &mu;s)

| Feature Name | Old Extractor (Wire Semantics) | New Extractor (Payload Semantics) | Delta (&Delta;) | Reference Definition (`BasicFlow.java`) |
| :--- | :---: | :---: | :---: | :--- |
| `Total Length of Fwd Packets` | 80.0 | **0.0** | -80.0 | `forwardBytes += packet.getPayloadBytes()` |
| `Total Length of Bwd Packets` | 40.0 | **0.0** | -40.0 | `backwardBytes += packet.getPayloadBytes()` |
| `Fwd Packet Length Min` | 40.0 | **0.0** | -40.0 | `fwdPktStats.addValue(packet.getPayloadBytes())` |
| `Fwd Packet Length Max` | 40.0 | **0.0** | -40.0 | `fwdPktStats.addValue(packet.getPayloadBytes())` |
| `Fwd Packet Length Mean` | 40.0 | **0.0** | -40.0 | `fwdPktStats.getMean()` |
| `Bwd Packet Length Mean` | 40.0 | **0.0** | -40.0 | `bwdPktStats.getMean()` |
| `Min Packet Length` | 40.0 | **0.0** | -40.0 | `flowLengthStats.getMin()` |
| `Max Packet Length` | 40.0 | **0.0** | -40.0 | `flowLengthStats.getMax()` |
| `Packet Length Mean` | 40.0 | **0.0** | -40.0 | `flowLengthStats.getMean()` |
| `Fwd Header Length` | 80.0 | **40.0** | -40.0 | `fHeaderBytes += packet.getHeaderBytes()` (2 pkts &times; 20B) |
| `Bwd Header Length` | 40.0 | **20.0** | -20.0 | `bHeaderBytes += packet.getHeaderBytes()` (1 pkt &times; 20B) |
| `min_seg_size_forward` | 40.0 | **20.0** | -20.0 | `min_seg_size_forward = packet.getHeaderBytes()` |
| `Down/Up Ratio` | 0.5 | **0.0** | -0.5 | `(int)(backward.size() / forward.size())` (1 // 2 = 0) |
| `Flow Bytes/s` | 4,000.0 | **0.0** | -4,000.0 | `(forwardBytes + backwardBytes) / duration` |
| `Average Packet Size` | 40.0 | **0.0** | -40.0 | `flowLengthStats.getSum() / packetCount()` |
| `Avg Fwd Segment Size` | 40.0 | **0.0** | -40.0 | `fwdPktStats.getSum() / forward.size()` |
| `Avg Bwd Segment Size` | 40.0 | **0.0** | -40.0 | `bwdPktStats.getSum() / backward.size()` |
| `Subflow Fwd Bytes` | 80.0 | **0.0** | -80.0 | `forwardBytes` |
| `Subflow Bwd Bytes` | 40.0 | **0.0** | -40.0 | `backwardBytes` |

#### Model Sensitivity on Fixture 1:
- **XGBoost Attack Probability:**
  - Old: $3.94 \times 10^{-6}$ (BENIGN)
  - New: $8.09 \times 10^{-7}$ (BENIGN)
- **PyTorch Tabular VAE Reconstruction Error ($MSE$):**
  - Old Extractor MSE: **`0.947866`**
  - New Extractor MSE: **`0.727991`** (&Delta; = **-0.219875**, a 23.2% error reduction on handshakes)

---

### Fixture 2: HTTP Transaction with Data Payload
- **Packets:** 8 packets (SYN, SYN-ACK, ACK, GET request 47B, HTTP 200 OK 59B, FIN-ACK, FIN-ACK, ACK)
- **Total Payload Bytes:** 106 B (47 B Fwd, 59 B Bwd)
- **Flow Duration:** 60 ms (60,000 &mu;s)

| Feature Name | Old Extractor (Wire Semantics) | New Extractor (Payload Semantics) | Delta (&Delta;) |
| :--- | :---: | :---: | :---: |
| `Total Length of Fwd Packets` | 207.0 | **47.0** | -160.0 |
| `Total Length of Bwd Packets` | 179.0 | **59.0** | -120.0 |
| `Fwd Packet Length Max` | 87.0 | **47.0** | -40.0 |
| `Fwd Packet Length Min` | 40.0 | **0.0** | -40.0 |
| `Bwd Packet Length Max` | 99.0 | **59.0** | -40.0 |
| `Bwd Packet Length Min` | 40.0 | **0.0** | -40.0 |
| `Min Packet Length` | 40.0 | **0.0** | -40.0 |
| `Max Packet Length` | 99.0 | **59.0** | -40.0 |
| `Fwd Header Length` | 160.0 | **80.0** | -80.0 |
| `Bwd Header Length` | 120.0 | **60.0** | -60.0 |
| `min_seg_size_forward` | 40.0 | **20.0** | -20.0 |
| `Down/Up Ratio` | 0.75 | **0.0** | -0.75 |
| `Flow Bytes/s` | 6,433.34 | **1,766.67** | -4,666.67 |

#### Model Sensitivity on Fixture 2:
- **XGBoost Attack Probability:**
  - Old: $0.201387$ (BENIGN)
  - New: $0.000525$ (BENIGN) &rarr; Massive reduction in false alert probability
- **PyTorch Tabular VAE Reconstruction Error ($MSE$):**
  - Old Extractor MSE: **`1.732540`**
  - New Extractor MSE: **`1.416335`** (&Delta; = **-0.316205**, an 18.3% error reduction)

---

### Fixture 3: Instantaneous Single-Packet Flow (0 &mu;s Duration)
- **Packets:** 1 UDP DNS packet (8 bytes payload)
- **Flow Duration:** 0.0 &mu;s

| Feature Name | Old Extractor (With Floor) | New Extractor (Safe Zero Duration) | Delta (&Delta;) |
| :--- | :---: | :---: | :---: |
| `Flow Bytes/s` | 36,000,000.0 | **0.0** | -36,000,000.0 |
| `Flow Packets/s` | 1,000,000.0 | **0.0** | -1,000,000.0 |
| `Fwd Packets/s` | 1,000,000.0 | **0.0** | -1,000,000.0 |
| `Fwd Header Length` | 28.0 | **8.0** | -20.0 |
| `min_seg_size_forward` | 28.0 | **8.0** | -20.0 |

---

### Fixture 4: Asymmetric Flow (Down/Up Integer Ratio Test)
- **Packets:** 8 Forward UDP packets, 5 Backward UDP packets
- **Calculation:** `5 // 8 = 0`

| Feature Name | Old Extractor (Float) | New Extractor (Integer) | Delta (&Delta;) |
| :--- | :---: | :---: | :---: |
| `Down/Up Ratio` | 0.625 | **0.0** | -0.625 |
| `min_seg_size_forward` | 28.0 | **8.0** | -20.0 |
| `Fwd Header Length` | 224.0 | **64.0** | -160.0 |
| `Bwd Header Length` | 140.0 | **40.0** | -100.0 |

---

## 3. Conclusions

1. The Stage C fixes remove the artificial wire-byte and IP-header offsets across all 21 dependent features without modifying any column names, ordering, or schema dimensions.
2. The VAE reconstruction error consistently shifts downwards toward the training baseline (e.g. dropping by 0.22 to 0.32 MSE points per flow).
3. XGBoost false-alarm probabilities drop substantially (e.g. from 20.1% to 0.05% on multi-packet HTTP flows).
4. All existing models, scalers, weights, thresholds, and academic test evaluations remain 100% intact.
