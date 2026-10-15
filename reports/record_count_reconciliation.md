# Cognisec AI — Record-Count Reconciliation Report

**Audit Subject:** Clarification of Test Sample Counts and Evaluation Volume  
**Investigation Target:** Reported `total_evaluated_records = 257,297` vs. Sum of Experiment Confusion Matrices (`125,088 + 172,209 = 297,297`)  
**Data Artifact:** `data/processed/test_data.npz` and `data/processed/partition_manifest.json`  
**Status:** **RESOLVED & MATHEMATICALLY VERIFIED**

---

## 1. Executive Summary & Root Cause

The investigation establishes that **both numbers are mathematically correct and describe distinct dimensions of the evaluation protocol**:

1. **`257,297` is the exact count of unique physical flow records** residing in the test partition (`data/processed/test_data.npz`).
2. **`297,297` is the total count of evaluation decisions** executed across Experiment A and Experiment B combined.
3. The numerical difference of exactly **`40,000` records** corresponds to the **shared benign control group** (`y == 0`), which was evaluated in *both* Experiment A (to compute false-alarm rates against known attacks) and Experiment B (to compute false-alarm rates against unseen zero-day attacks).

There is **zero discrepancy, zero missing data, and zero phantom evaluation**. The apparent discrepancy arose solely from labeling `len(y_test)` as `"total_evaluated_records"` in the top-level JSON without annotating the experimental reuse of the benign control baseline.

---

## 2. Partition Structure and Reconciliation Table

| Metric / Partition Slice | Benign Count (`y=0`) | Attack Count (`y=1`) | Total Slice Size | Overlap with Other Slices | Source Code & Array Slice |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Physical Test Partition (`test_data.npz`)** | **40,000** | **217,297** | **257,297** | None (100% unique physical rows) | `test_data.npz` (lines 61–65 of `run_experiments.py`) |
| **Experiment A (Known Attacks)** | **40,000** | **85,088** | **125,088** | 40,000 Benign shared with Exp B | `np.isin(families_test, ['BENIGN', 'DoS', 'PortScan', 'Brute_Force', 'Heartbleed'])` |
| **Experiment B (Held-Out Unseen Attacks)** | **40,000** | **132,209** | **172,209** | 40,000 Benign shared with Exp A | `np.isin(families_test, ['BENIGN', 'DDoS', 'Web_Attack', 'Bot', 'Infiltration'])` |
| **Combined Benchmark Evaluations (Exp A + Exp B)** | **80,000** (40k x 2) | **217,297** | **297,297** | 40,000 duplicated benign evaluations | Sum of Confusion Matrices |

### Reconciliation Equation
$$\text{Total Benchmark Evaluations} = \text{Exp A Evaluations} (125,088) + \text{Exp B Evaluations} (172,209) = 297,297$$
$$\text{Unique Physical Test Records} = 297,297 - \text{Shared Benign Control Group} (40,000) = 257,297$$

---

## 3. Confusion Matrix Verification

### Experiment A: Known Attack Family Benchmark
* Ground Truth: 40,000 Benign, 85,088 Known Attacks (Total = 125,088)
* **XGBoost:**
  * $\text{True Negatives (TN)} = 39,961$
  * $\text{False Positives (FP)} = 39$
  * $\text{Benign Check: } \text{TN} + \text{FP} = 39,961 + 39 = 40,000$ (Exact match)
  * $\text{False Negatives (FN)} = 8$
  * $\text{True Positives (TP)} = 85,080$
  * $\text{Attack Check: } \text{FN} + \text{TP} = 8 + 85,080 = 85,088$ (Exact match)
  * $\text{Sum} = 39,961 + 39 + 8 + 85,080 = 125,088$ (100% verified)
* **Tabular VAE:**
  * $\text{TN} = 39,628$ | $\text{FP} = 372$ | $\text{TN} + \text{FP} = 40,000$ (Exact match, 0.93% test FPR)
  * $\text{FN} = 63,058$ | $\text{TP} = 22,030$ | $\text{FN} + \text{TP} = 85,088$ (Exact match)
  * $\text{Sum} = 39,628 + 372 + 63,058 + 22,030 = 125,088$ (100% verified)

### Experiment B: Held-Out Zero-Day Benchmark
* Ground Truth: 40,000 Benign, 132,209 Unseen Attacks (Total = 172,209)
* **XGBoost:**
  * $\text{TN} = 39,961$ | $\text{FP} = 39$ | $\text{TN} + \text{FP} = 40,000$ (Identical benign predictions)
  * $\text{FN} = 49,574$ | $\text{TP} = 82,635$ | $\text{FN} + \text{TP} = 132,209$ (Exact match)
  * $\text{Sum} = 39,961 + 39 + 49,574 + 82,635 = 172,209$ (100% verified)
* **Tabular VAE:**
  * $\text{TN} = 39,628$ | $\text{FP} = 372$ | $\text{TN} + \text{FP} = 40,000$ (Identical benign predictions)
  * $\text{FN} = 110,268$ | $\text{TP} = 21,941$ | $\text{FN} + \text{TP} = 132,209$ (Exact match)
  * $\text{Sum} = 39,628 + 372 + 110,268 + 21,941 = 172,209$ (100% verified)

---

## 4. Disaggregated Attack Breakdown

Sum of Attack Classes across the physical test split:
1. **Known Attacks:**
   - DoS: 50,645
   - PortScan: 31,666
   - Brute_Force: 2,774
   - Heartbleed: 3
   - **Subtotal Known Attacks:** 85,088
2. **Held-Out Attacks:**
   - DDoS: 128,027
   - Web_Attack: 2,180
   - Bot: 1,966
   - Infiltration: 36
   - **Subtotal Held-Out Attacks:** 132,209
3. **Total Attacks:** $85,088 + 132,209 = 217,297$
4. **Total Unique Test Records:** $40,000 \text{ (Benign)} + 217,297 \text{ (Attacks)} = 257,297$

---

## 5. Resolution & JSON Enhancement

To eliminate ambiguity for future academic reviewers and viva examiners:
1. Updated [results/master_experiment_results.json](file:///c:/Users/Santhoskrishna/Documents/Cognisec_AI/results/master_experiment_results.json) to explicitly distinguish:
   - `"total_unique_test_records": 257297`
   - `"total_benchmark_evaluations": 297297`
   - `"shared_benign_control_records": 40000`
   - Retained `"total_evaluated_records": 257297` for backwards compatibility with existing UI charts.
2. Verified that all metric calculations, confusion matrix values, and precision/recall formulas derive from the exact ground-truth subsets.
