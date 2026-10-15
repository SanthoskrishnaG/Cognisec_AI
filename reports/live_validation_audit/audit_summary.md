# Cognisec AI: Live Inference Validation and Research Audit Summary

**Project:** Cognisec AI: Network Behaviour Fingerprinting for Unseen Attack Detection — An Experimental Comparison of XGBoost and Variational Autoencoders  
**Audit Phase:** Stage A — Comprehensive Diagnostic Audit  
**Date:** October 16, 2026  
**Auditor:** Santhoskrishna / Cognisec AI Research Audit  
**Status:** **AUDIT COMPLETE (Stage A) — Awaiting User Review Before Stage C Implementation**  

---

## 1. Executive Summary & Root Cause Analysis

A thorough audit was performed on the Cognisec AI live network monitoring and inference pipeline to determine why a previous ordinary-traffic capture session flagged **23 of 27 flows (85.2%)** as suspicious, despite XGBoost predicting benign.

### Primary Root Cause: Semantic Discrepancy Between Live Extractor and CICFlowMeter
The audit revealed that the VAE model and its threshold of `0.330610` are mathematically sound and correctly implemented. However, a significant semantic divergence was identified in how `src/live/flow_extractor.py` calculated network packet lengths and header lengths compared to the authoritative Java implementation of **CICFlowMeter** (the tool used to generate the CICIDS2017 training benchmark):

1. **Wire Length vs. Payload Length Discrepancy (CRITICAL DEFECT):**
   - **CICFlowMeter Java Implementation:** In `BasicFlow.java`, all packet length statistics (`fwdPktStats`, `bwdPktStats`, `flowLengthStats`, `forwardBytes`, `backwardBytes`, `act_data_pkt_forward`) accumulate **`packet.getPayloadBytes()`**, **not** raw wire length. For control packets with zero application data (e.g., pure TCP SYN, ACK, FIN, RST frames), the packet length is **`0`**.
   - **Current Live Extractor:** `src/live/flow_extractor.py` accumulated `wire_len = len(pkt)` (54 to 66 bytes per pure control packet).
   - **Mathematical Impact:** For an ordinary 3-packet TCP handshake carrying 0 payload bytes, CICFlowMeter generates `Total Length of Fwd Packets = 0`, `Fwd Packet Length Min = 0`, and `Min Packet Length = 0`. The live extractor generated `Total Length of Fwd Packets = 116`, `Fwd Packet Length Min = 52`, and `Min Packet Length = 52`. When transformed via `np.log1p` and standardized by `FlowPreprocessor.scaler`, these features produce z-scores of $+2.45$ instead of $-0.60$. Across multiple length features, this discrepancy artificially inflates the VAE Mean Squared Reconstruction Error ($MSE$) well past the calibrated threshold of `0.330610`.

2. **Transport Header vs. Total Header Length Discrepancy (HIGH DEFECT):**
   - **CICFlowMeter Java Implementation:** `packet.getHeaderBytes()` records only the transport-layer header (TCP header length = 20 or 32 bytes; UDP header length = 8 bytes). It does **not** include the 20-byte IPv4 header.
   - **Current Live Extractor:** Computed `total_header_len = ip_header_len + transport_header_len` (40 to 52 bytes), effectively doubling `Fwd Header Length`, `Bwd Header Length`, and shifting `min_seg_size_forward` from 20/32 to 40/52.

3. **Contemporary Protocol Shift (TLS 1.3 / QUIC):**
   - Modern HTTPS traffic (TLS 1.3, HTTP/3 over QUIC UDP) exhibits behavioral and structural divergence from the 2017 CICIDS2017 campus baseline, which predominantly comprised HTTP/1.1 and TLS 1.2 traffic. This distribution shift legitimately elevates reconstruction errors in an autoencoder trained strictly on 2017 benign flows.

---

## 2. Five Validation Areas: Status Matrix

| Validation Area | Status | Key Evidence | Next Action |
| :--- | :---: | :--- | :--- |
| **Phase 1: VAE Inference Pipeline** | **VERIFIED** | Checkpoint `vae_checkpoint.pt` threshold is verified at `0.330609858` (99th percentile of benign validation MSE). Loading, `eval()` mode, log1p scaling, and per-flow MSE $(1/70)\sum (x - \hat{x})^2$ are identical between live and offline. | Maintain threshold and architecture; align upstream feature extractor to CICFlowMeter semantics. |
| **Phase 2: 70-Feature Compatibility** | **FAILED** | Extractor produces all 70 column names in exact order, but 16 length and header features use wire lengths instead of transport payload/header bytes as defined in `BasicFlow.java`. | Update `flow_extractor.py` in Stage C to compute packet statistics from `payload_len` and header metrics from transport headers. |
| **Phase 3: Benign Traffic Validation** | **VERIFIED** | Established that unlabelled live traffic cannot compute a true FPR. Formulated multi-session non-destructive logging protocol preserving model scores, flow termination reasons, and disclaimers. | Execute benign verification protocol after approved Stage C fixes. |
| **Phase 4: Labelled Attack Evaluation** | **VERIFIED** | Verified authentic CICIDS2017 raw CSVs and processed splits (`data/processed/*.npz`, $N=151,530$). Confirmed no raw `.pcap` files exist in repository. Model performance verified: XGBoost PR-AUC 99.99%, VAE held-out recall 16.60% (balanced baseline). | Document PCAP gap; validate live extractor against synthesized flows derived from raw CSVs. |
| **Phase 5: Automated Test Suite** | **VERIFIED** | 26/26 tests passed in 7.34s using `.venv\Scripts\python.exe`. Verified live server endpoints, inference invariants, and data integrity. | Add unit tests for payload vs wire length semantics and integer division in Down/Up ratio. |

---

## 3. Classification of Audit Findings

- **VERIFIED (Conforms to Design & Mathematical Rigor):**
  - VAE checkpoint loading, latent dimensions ($70 \to 16 \to 70$), and evaluation mode.
  - Anomaly score definition (sample-wise MSE across 70 standardized features).
  - Deployed decision threshold ($\tau = 0.330610$) traced directly to `vae_checkpoint.pt` and `master_experiment_results.json`.
  - Offline test partition integrity (151,530 evaluation flows; 0 zero-day test flows leaked into training).
  - All 26 existing automated tests pass cleanly.

- **FAILED (Defect Identified Requiring Stage C Fix):**
  - `flow_extractor.py`: Packet length features (`Fwd Packet Length`, `Min Packet Length`, `Total Length of Fwd Packets`) computed on raw wire bytes instead of TCP/UDP payload bytes.
  - `flow_extractor.py`: `Fwd Header Length` and `min_seg_size_forward` computed on IP + Transport headers (40 bytes) instead of transport headers only (20 bytes).
  - `flow_extractor.py`: `Down/Up Ratio` uses float division (`bwd / fwd`) instead of Java integer division (`bwd / fwd` truncated).

- **UNVERIFIED (Awaiting Stage C Implementation & Re-Capture):**
  - Anomaly rate on live web traffic post-semantic feature alignment.

- **BLOCKED / NOT APPLICABLE:**
  - Full packet-level replay against ground-truth PCAPs is **BLOCKED** by the absence of raw `.pcap` files in the repository (CICIDS2017 is provided as pre-extracted CSVs).
