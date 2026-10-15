# Phase 3: Benign-Traffic False-Positive Validation Protocol & Operational Plan

**Document:** `reports/live_validation_audit/benign_traffic_validation_plan.md`  
**Scope:** Local Evaluation Procedure for Ordinary Traffic Captured from Authorized Host Interfaces  
**Auditor:** Santhoskrishna / Cognisec AI Research Audit  
**Status:** **AUDIT PLAN COMPLETE (Stage A) — Execution Scheduled Post-Fix Approval**  

---

## 1. Methodological Clarification: Anomaly Percentage vs. False-Positive Rate

In academic and operational network security research, conflating unlabelled alert counts with mathematical error rates is an invalid practice:

```text
+---------------------------------------------------------------------------------------------------+
|  1. Observed Anomaly Percentage = (Number of Flows where Score >= Threshold) / (Total Live Flows)   |
|     - Operates on UNLABELLED live network traffic.                                               |
|     - Represents operational alert volume and distribution shift, NOT an error rate.             |
+---------------------------------------------------------------------------------------------------+
                                                  ≠
+---------------------------------------------------------------------------------------------------+
|  2. Confirmed False Positive Rate (FPR) = FP / (FP + TN)                                          |
|     - Requires VERIFIED GROUND-TRUTH BENIGN LABELS (y_true = 0).                                  |
|     - Cannot be established on unlabelled live laptop traffic without manual protocol packet-by-   |
|       packet forensic verification.                                                              |
+---------------------------------------------------------------------------------------------------+
                                                  ≠
+---------------------------------------------------------------------------------------------------+
|  3. Benign-Flow Rejection Rate = (Flagged Benign Flows) / (Total Verified Benign Flows)           |
|     - Evaluated on controlled, verified benign corpora (such as CICIDS2017 Monday benign captures)|
|     - The VAE threshold of 0.330610 enforces a calibrated rejection rate ceiling of <= 1.0%.     |
+---------------------------------------------------------------------------------------------------+
```

> [!IMPORTANT]
> Unlabelled live traffic **cannot** establish a mathematically true False Positive Rate ($FPR$). All live monitoring outputs must be explicitly reported as **Observed Anomaly Percentages** or **Model Prediction Frequencies**.

---

## 2. Multi-Session Benign Traffic Capture Protocol

To characterize model behavior and alert volume across diverse real-world conditions without generating attacks or scanning third parties, three standardized, non-destructive capture sessions are defined:

### 2.1 Standardized Evaluation Sessions

| Session ID | Network Scenario | Authorized Target | Target Packets / Flows | Expected Traffic Profile |
| :---: | :--- | :--- | :---: | :--- |
| **SESS-01-LOOPBACK** | Local Host IPC / API Calls | `Npcap Loopback Adapter` (`127.0.0.1`) | 100 flows | Controlled REST API queries, high throughput, zero external latency |
| **SESS-02-IDLE-WIFI** | Passive Idle Workstation | Host Wi-Fi (`10.145.180.196`) | 50 flows | OS telemetry, NTP time sync, background DNS queries, zero active browser tabs |
| **SESS-03-ACTIVE-WEB** | Ordinary Web Browsing | Host Wi-Fi (`10.145.180.196`) | 100 flows | Standard HTTPS (Wikipedia, Google, GitHub), TLS 1.3 / TCP handshakes, media streaming |

---

## 3. Mandatory Safeguards & Operational Requirements

1. **Passive Sniffing Exclusively:** Capture operates in promiscuous/passive mode on the local machine only. No probe frames, port scans, or packet injections are ever transmitted.
2. **Immediate Payload Discard:** Layer-7 payload contents (application data, passwords, HTTP bodies) are scrubbed immediately after recording header length and payload byte size.
3. **Zero Raw Traffic Persistence:** No raw `.pcap` files containing sensitive user traffic are stored on disk.
4. **Structured JSON Telemetry:** Flows are exported to `reports/live_validation_audit/sessions/{session_id}.json` with complete per-flow metadata:
   - Unique Flow ID (`src_ip:src_port-dst_ip:dst_port_proto`)
   - Flow Termination Reason (`TCP_FIN_BOTH`, `TCP_RST`, `IDLE_TIMEOUT`, `ACTIVE_TIMEOUT`, `CAPTURE_STOPPED`)
   - Flow Duration ($\mu s$ and seconds)
   - Total Forward & Backward Packets and Bytes
   - XGBoost Probability, Prediction (`Benign` vs `Attack`), and Decision Threshold (`0.5200`)
   - PyTorch Tabular VAE Anomaly Score (MSE), Decision (`Normal` vs `Anomaly`), and Calibrated Threshold (`0.330610`)
   - Joint Classification Tag (`High-Confidence Attack`, `Signature Flag`, `Unseen Anomaly`, `Normal Behavior`)
   - Explicit Disclaimer Tag: `"Model Prediction Only (No Ground-Truth Label)"`

---

## 4. Flow Stratification: Complete vs. Incomplete Flows

In live networks, network flows terminate under different conditions:
1. **Cleanly Terminated Flows:** Complete TCP sessions observing two-way `FIN-ACK` handshakes or explicit `RST` termination.
2. **Inactivity Expired Flows:** Flows that went idle for $> 15.0$ seconds without an explicit FIN/RST (common for long-lived keep-alive HTTP/2 connections).
3. **Flushed / Incomplete Flows:** In-flight flows forcefully flushed when the user clicks "Stop Capture".

The validation procedure separately analyzes:
- The anomaly distribution of **cleanly terminated flows** versus **flushed incomplete flows**.
- This isolates whether incomplete TCP states artificially elevate reconstruction error relative to closed flows.

---

## 5. Environmental Distribution Shift Analysis

When comparing live 2026 workstation traffic against the 2017 CICIDS2017 benchmark, several authentic domain shifts must be acknowledged:
- **TLS 1.3 vs. TLS 1.2:** TLS 1.3 encrypts certificate handshakes and reduces round trips, changing packet size and timing distributions.
- **QUIC / HTTP/3 over UDP:** Modern Google, Cloudflare, and YouTube traffic utilizes QUIC over UDP port 443 with large initial packet sizes (1200+ bytes), a protocol that did not exist in the 2017 dataset.
- **Autoencoder Sensitivity:** Because the Tabular VAE evaluates whether a flow belongs to the 2017 campus benign manifold, novel modern cipher handshakes produce larger reconstruction errors. This is scientifically valid behavior reflecting **unseen protocol shift**, not model malfunction.
