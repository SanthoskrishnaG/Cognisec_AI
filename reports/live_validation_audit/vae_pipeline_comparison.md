# Phase 1: VAE Inference Pipeline Line-by-Line Comparison & Audit

**Document:** `reports/live_validation_audit/vae_pipeline_comparison.md`  
**Scope:** Comprehensive Trace from Packet Capture to Final VAE Decision  
**Auditor:** Santhoskrishna / Cognisec AI Research Audit  
**Status:** **AUDIT COMPLETE**  

---

## 1. Line-by-Line Pipeline Trace

```text
[ Live Network Flow ] ──> [ flow_extractor.py ] ──> [ FlowPreprocessor.transform() ] ──> [ TabularVAE.forward() ] ──> [ MSE Anomaly Score ] ──> [ Decision: score >= 0.330610 ]
```

### 1.1 Step 1: Live Packet Parsing & Feature Assembly
- **Live File & Method:** `src/live/flow_extractor.py` &rarr; `BidirectionalFlow.to_features_dict()`
- **Offline Baseline:** `data/raw/*.csv` generated via CICFlowMeter (`BasicFlow.java`).
- **Audit Findings:**
  - `to_features_dict()` generates a dictionary containing all 70 features in identical casing and key naming as `FlowPreprocessor.feature_columns`.
  - **Divergence:** As documented in Phase 2, `to_features_dict()` utilizes `PacketRecord.wire_len` instead of `PacketRecord.payload_len`, and combines IP + Transport header lengths instead of transport header length alone.

### 1.2 Step 2: DataFrame Conversion & Column Alignment
- **Live File & Method:** `src/live/live_manager.py` &rarr; Lines 220–224:
  ```python
  features_dict = flow.to_features_dict()
  df_flow = pd.DataFrame([features_dict])
  X_unscaled, X_scaled, _, _ = self.preprocessor.transform(df_flow)
  ```
- **Offline Baseline:** `src/api/server.py` &rarr; Lines 523, 581:
  ```python
  _, X_scaled, _, raw_labels = preprocessor.transform(df)
  ```
- **Audit Comparison:** **IDENTICAL**. The live pipeline feeds the DataFrame into the exact same `preprocessor.transform()` method as the offline batch upload route.

### 1.3 Step 3: Imputation & Variance-Stabilizing Log Transformation
- **Pipeline Implementation:** `src/data/preprocessing.py` &rarr; `FlowPreprocessor.transform()`:
  ```python
  # Align columns and impute with training medians
  imputed_df = pd.DataFrame(index=df.index)
  for col in self.feature_columns:
      if col in df.columns:
          imputed_df[col] = df[col].fillna(self.median_values[col])
      else:
          imputed_df[col] = self.median_values[col]

  X_unscaled = imputed_df[self.feature_columns].values.astype(np.float32)
  X_log = np.log1p(np.maximum(0, X_unscaled))
  X_scaled = self.scaler.transform(X_log).astype(np.float32)
  X_scaled = np.clip(X_scaled, -5.0, 5.0)
  ```
- **Audit Findings:**
  - Non-negative clipping `np.maximum(0, X_unscaled)` is applied before `np.log1p()`.
  - Missing values are imputed with `self.median_values` (learned strictly on the training partition).
  - Standard scaling: $z = (X_{\log} - \mu) / \sigma$.
  - Extreme values are clipped to $[-5.0, 5.0]$ to prevent numerical overflow in the deep network.
  - **Verdict:** **IDENTICAL & PRESERVED**.

### 1.4 Step 4: VAE Model Architecture & Checkpoint Loading
- **Model Architecture:** `src/models/vae_model.py` &rarr; `TabularVAE`:
  - Input Dimension: 70
  - Encoder Backbone: Linear(70, 64) &rarr; BatchNorm1d(64) &rarr; LeakyReLU(0.2) &rarr; Dropout(0.1) &rarr; Linear(64, 32) &rarr; BatchNorm1d(32) &rarr; LeakyReLU(0.2) &rarr; Dropout(0.1)
  - Latent Bottleneck: $\mu \in \mathbb{R}^{16}$, $\log \sigma^2 \in \mathbb{R}^{16}$
  - Decoder: Linear(16, 32) &rarr; BatchNorm1d(32) &rarr; LeakyReLU(0.2) &rarr; Dropout(0.1) &rarr; Linear(32, 64) &rarr; BatchNorm1d(64) &rarr; LeakyReLU(0.2) &rarr; Dropout(0.1) &rarr; Linear(64, 70)
- **Checkpoint Verification:** `artifacts/models/vae_checkpoint.pt`:
  - Input Dim: 70
  - Latent Dim: 16
  - Encoder Hidden: [64, 32]
  - Decoder Hidden: [32, 64]
  - Saved Threshold: `0.33060985803604126`
  - Target FPR: 1.0%
- **Evaluation Mode:**
  - `compute_anomaly_scores` explicitly invokes `self.model.eval()`:
    ```python
    self.model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            ...
    ```
  - In evaluation mode:
    1. Dropout is deactivated ($p=0.0$).
    2. BatchNorm uses stored running mean and variance ($\mu_{\text{running}}, \sigma^2_{\text{running}}$).
    3. The reparameterization function deterministically returns the latent mean:
       ```python
       def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
           if self.training:
               ...
           return mu
       ```
  - **Verdict:** **VERIFIED & CORRECT**.

### 1.5 Step 5: Reconstruction Loss & Anomaly Score Definition
- **Definition:** `src/models/vae_model.py` &rarr; `compute_anomaly_scores`:
  ```python
  sq_errors = torch.mean((batch_x - recon) ** 2, dim=1)
  scores.append(sq_errors.cpu().numpy())
  ```
- **Mathematical Formula:**
  $$s(\mathbf{x}) = \frac{1}{70} \sum_{j=1}^{70} \left(x_j - \hat{x}_j\right)^2$$
- **Audit Findings:**
  - Reduction: Feature-wise mean across all 70 features (`dim=1`).
  - Unit: Mean squared standardized reconstruction error ($MSE$).
  - **Verdict:** **IDENTICAL** across offline training, offline validation, and live monitoring.

### 1.6 Step 6: Threshold Source & Calibration Origin
- **Source Code:** `src/models/vae_model.py` &rarr; `calibrate_threshold`:
  ```python
  val_scores = self.compute_anomaly_scores(X_val_benign)
  percentile = 100.0 - target_fpr
  self.threshold = float(np.percentile(val_scores, percentile))
  ```
- **Empirical Derivation:**
  - Calibration Dataset: Benign validation partition ($N = 45,459$ benign flows).
  - Target False Positive Rate ($FPR$): Fixed at $1.0\%$ (operational false-alarm ceiling).
  - Empirical 99.0th percentile of benign validation reconstruction errors:
    $$\tau = \text{Percentile}\left(\left\{s(\mathbf{x}_{\text{val, benign}})\right\}, 99.0\right) = 0.330610$$
  - Threshold Verification: The value `0.33060985803604126` saved in `vae_checkpoint.pt` corresponds directly to this 99th percentile calculation.
  - **Verdict:** **VERIFIED**. The live threshold is genuine and correctly sourced.

### 1.7 Step 7: Joint Decision & Threat Level Synthesis
- **Implementation:** `src/live/live_manager.py` &rarr; Lines 237–250:
  ```python
  if xgb_is_attack and vae_is_anomaly:
      joint_decision = "High-Confidence Attack (Both Flagged)"
      decision_level = "critical"
  elif xgb_is_attack and not vae_is_anomaly:
      joint_decision = "Supervised Pattern Detected (XGBoost Only)"
      decision_level = "warning"
  elif not xgb_is_attack and vae_is_anomaly:
      joint_decision = "Unseen / Behavioral Anomaly (VAE Only)"
      decision_level = "anomaly"
  else:
      joint_decision = "Normal Network Behavior"
      decision_level = "normal"
  ```
- **Audit Findings:**
  - Correctly reflects the core thesis of Cognisec AI: XGBoost flags known signature attacks, while the VAE flags behavioral anomalies.
  - Correctly tags records with: `"disclaimer": "Model Prediction Only (No Ground-Truth Label)"`.

---

## 2. Detailed Findings & Severity Matrix

| Finding ID | Component | Description & Evidence | Severity | Proposed Fix |
| :---: | :--- | :--- | :---: | :--- |
| **VAE-01** | `flow_extractor.py` | **Packet Length Semantics Mismatch:** Length statistics accumulated `p.wire_len` instead of `p.payload_len`. In CICFlowMeter `BasicFlow.java`, `fwdPktStats` uses `packet.getPayloadBytes()`. | **CRITICAL** | In Stage C, accumulate `payload_len` for forward/backward packet length statistics. |
| **VAE-02** | `flow_extractor.py` | **Header Length Semantics Mismatch:** `fwd_header_len` and `min_seg_size_forward` accumulated IP + Transport headers (40 bytes) instead of transport headers only (20 bytes). | **HIGH** | In Stage C, record transport layer header length (`transport_header_len`) for header features. |
| **VAE-03** | `flow_extractor.py` | **Down/Up Ratio Semantics:** Used continuous float division `bwd / fwd` instead of Java integer division `(int)(bwd / fwd)` as performed in `BasicFlow.java:243`. | **MEDIUM** | In Stage C, cast to `int(tot_bwd_pkts / max(tot_fwd_pkts, 1))` to mirror CICFlowMeter integer division. |
| **VAE-04** | `flow_extractor.py` | **Zero-Duration Single-Packet Rates:** For 0-duration flows, rates were computed using `eff_duration_sec = 1e-6` ($1 \mu s$), yielding 1,000,000 pkts/s instead of 0.0. | **LOW** | In Stage C, emit 0.0 for flow rates when total duration is 0 and packet count is 1. |
| **VAE-05** | `live_manager.py` | **Threshold and Architecture Match:** Deployed threshold `0.330610` and reconstruction loss definition are verified 100% compatible with training checkpoints. | **INFO** | Retain threshold `0.330610` without alteration. |
