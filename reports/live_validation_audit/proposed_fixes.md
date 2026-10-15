# Proposed Stage C Implementation & Fix Plan

**Document:** `reports/live_validation_audit/proposed_fixes.md`  
**Audit Context:** Stage A Findings & Stage B Review Proposal  
**Auditor:** Santhoskrishna / Cognisec AI Research Audit  
**Status:** **PROPOSED — AWAITING USER APPROVAL (Stage B Review)**  

---

## 1. Principles of Proposed Modifications

1. **Strict Non-Destructive Invariance:**
   - **DO NOT** modify `artifacts/models/xgboost_model.json`.
   - **DO NOT** modify `artifacts/models/vae_checkpoint.pt`.
   - **DO NOT** modify `artifacts/models/flow_preprocessor.joblib`.
   - **DO NOT** alter the calibrated VAE decision threshold (`0.330610`) or XGBoost threshold (`0.5200`).
   - **DO NOT** alter existing CSV upload routes or research comparison views.
2. **Targeted Upstream Semantic Alignment:**
   - All proposed fixes reside solely in `src/live/flow_extractor.py`.
   - The goal is to bring the live packet feature extractor into exact mathematical conformity with the reference Java implementation of **CICFlowMeter** (`BasicFlow.java`), which generated the training data.

---

## 2. Proposed Code Changes (Stage C)

### Fix 1: Packet Length Statistics (FIND-01)
- **File:** `src/live/flow_extractor.py` &rarr; `PacketRecord` & `to_features_dict()`
- **Current Behavior:**
  ```python
  fwd_lens = [p.wire_len for p in fwd]
  bwd_lens = [p.wire_len for p in bwd]
  all_lens = [p.wire_len for p in all_pkts]
  ```
- **Proposed Correction:**
  In accordance with `BasicFlow.java` lines 137 and 151 (`fwdPktStats.addValue(packet.getPayloadBytes())`):
  ```python
  fwd_lens = [p.payload_len for p in fwd]
  bwd_lens = [p.payload_len for p in bwd]
  all_lens = [p.payload_len for p in all_pkts]
  tot_len_fwd = sum(fwd_lens)
  tot_len_bwd = sum(bwd_lens)
  ```
  *(Note: Non-zero payload bytes are tracked; pure control packets with no data correctly register a length of 0 bytes).*

### Fix 2: Transport Header Length Statistics (FIND-02)
- **File:** `src/live/flow_extractor.py` &rarr; `PacketRecord` & `to_features_dict()`
- **Current Behavior:**
  ```python
  total_header_len = ip_header_len + transport_header_len  # (e.g. 20 + 20 = 40)
  fwd_header_len = sum(p.header_len for p in fwd)
  min_seg_fwd = min((p.header_len for p in fwd), default=0.0)
  ```
- **Proposed Correction:**
  In accordance with `BasicFlow.java` lines 148 and 152 (`fHeaderBytes = packet.getHeaderBytes()`, where `getHeaderBytes()` is TCP/UDP header length only):
  - Store `transport_header_len` in `PacketRecord.header_len`.
  - Compute `fwd_header_len = sum(p.header_len for p in fwd)` (e.g. 20 bytes for standard TCP, 32 bytes for TCP with options, 8 bytes for UDP).
  - Compute `min_seg_fwd = min((p.header_len for p in fwd), default=0.0)` (e.g. 20 bytes).

### Fix 3: Integer Division for Down/Up Ratio (FIND-03)
- **File:** `src/live/flow_extractor.py` &rarr; `to_features_dict()`
- **Current Behavior:**
  ```python
  down_up_ratio = float(tot_bwd_pkts) / max(float(tot_fwd_pkts), 1.0)
  ```
- **Proposed Correction:**
  In accordance with `BasicFlow.java:243` (`(double)(this.backward.size() / this.forward.size())`):
  ```python
  down_up_ratio = float(int(tot_bwd_pkts / max(tot_fwd_pkts, 1)))
  ```

### Fix 4: Zero-Duration Single-Packet Rate Safety (FIND-04)
- **File:** `src/live/flow_extractor.py` &rarr; `to_features_dict()`
- **Proposed Correction:**
  ```python
  if duration_sec <= 0.0:
      flow_bytes_s = 0.0
      flow_pkts_s = 0.0
      fwd_pkts_s = 0.0
      bwd_pkts_s = 0.0
  else:
      flow_bytes_s = tot_bytes / duration_sec
      flow_pkts_s = tot_pkts / duration_sec
      fwd_pkts_s = tot_fwd_pkts / duration_sec
      bwd_pkts_s = tot_bwd_pkts / duration_sec
  ```

---

## 3. Anticipated Impact on Live VAE Anomaly Scoring

Applying these upstream fixes:
1. Eliminates the artificial z-score inflation ($+2.45 \to -0.60$) on ordinary TCP handshakes.
2. Resolves the false elevation of `min_seg_size_forward` ($40 \to 20$) and `Fwd Header Length`.
3. Reduces baseline VAE reconstruction errors for ordinary clean TCP/UDP handshakes from $\approx 1.05$ to well below the operational threshold of $0.330610$.
4. **Does not alter the academic evaluation:** Offline test metrics on authentic CICIDS2017 data remain 100% identical.

---

## 4. Proposed Stage C & D Execution Workflow

1. **User Review:** Review this proposed fix plan.
2. **Implement Minimal Edits:** Apply Fixes 1–4 to `src/live/flow_extractor.py`.
3. **Add Semantic Unit Tests:** Add regression tests in `tests/test_live_monitoring.py` confirming that zero-payload packets yield length = 0 and transport-only header sizes.
4. **Run Full Regression Suite:** Confirm all 26 existing tests + new unit tests pass (`pytest tests/ -v`).
5. **Execute Benign Validation Capture:** Run a fresh 3-session benign capture and record anomaly score distributions in `reports/live_validation_audit/benign_validation_results.md`.
