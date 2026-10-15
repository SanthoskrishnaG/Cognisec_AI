# Cognisec AI — Statistical Validation Report

**Investigation Target:** Paired Comparative Metrics, Confidence Intervals, and Hypothesis Testing  
**Models Compared:** Supervised XGBoost Classifier vs. Unsupervised PyTorch Tabular VAE  
**Dataset:** Authentic CICIDS2017 Test Partitions (`test_data.npz`)  
**Methodology:** Paired Non-Parametric Bootstrap Resampling ($B=1,000$) & McNemar's Test with Continuity Correction  
**Status:** **AUDITED, RECALCULATED, & FORMALLY VERIFIED**

---

## 1. Executive Summary & Anomaly Audit

The original `master_experiment_results.json` reported:
- Known Experiment: $\Delta = 0.5898$, $95\% \text{ CI } [0.5859, 0.5938]$, `p_value_approx = 0.0`
- Held-Out Experiment: $\Delta = 0.4591$, $95\% \text{ CI } [0.4568, 0.4619]$, `p_value_approx = 0.0`

### Audit Findings
1. **Metric Definition Identified:**
   - In Experiment A, $\Delta$ represents the **paired difference in $F_1$-score**: $\Delta F_1 = F_1(\text{XGBoost}) - F_1(\text{VAE})$.
   - In Experiment B, $\Delta$ represents the **paired difference in Detection Recall**: $\Delta \text{Recall} = \text{Recall}(\text{XGBoost}) - \text{Recall}(\text{VAE})$.
2. **Observations are Genuinely Paired:**
   - Both models evaluated the exact identical feature vectors on the exact identical row indices for each test partition slice.
3. **P-Value Flaw Corrected:**
   - The reported `p_value_approx = 0.0` occurred because empirical bootstrap counting `np.mean(diffs <= 0)` observed zero resamples crossing zero out of $B=300$.
   - **Correction:** In rigorous academic statistics, a non-parametric empirical bootstrap cannot report $p = 0.0$. The mathematically correct bound is **$p < 0.001$**, corroborated by parametric asymptotic testing (McNemar's $\chi^2 > 60,000$, $p < 10^{-300}$).

---

## 2. Statistical Methodology Selection

### A. Paired Empirical Bootstrap ($B = 1,000$, Random Seed 42)
- **Objective:** Establish distribution-free 95% confidence intervals for performance differences on paired test predictions without making normality assumptions.
- **Resampling Unit:** Flow records with replacement ($N_{\text{Exp A}} = 125,088$; $N_{\text{Exp B}} = 172,209$).
- **Percentile Interval:** Empirical quantiles at $\alpha/2 = 0.025$ and $1 - \alpha/2 = 0.975$.

### B. McNemar’s Discordance Test (with Continuity Correction)
- **Objective:** Evaluate whether marginal proportions of classification errors differ significantly between the two models on the same evaluation flows.
- **Contingency Table Formulation:**
  - $a$: Both models correct.
  - $b$: XGBoost correct, VAE incorrect.
  - $c$: XGBoost incorrect, VAE correct.
  - $d$: Both models incorrect.
- **Test Statistic:** $\chi^2 = \frac{(|b - c| - 1)^2}{b + c} \sim \chi^2_1$.

---

## 3. Recalculated Experimental Comparisons

### Experiment A: Known Attack Benchmark ($N = 125,088$)
- **Compared Metric:** $F_1$-Score
  - Point Estimates: $\text{XGBoost } F_1 = 0.9997$ | $\text{VAE } F_1 = 0.4099$
  - Effect Size: $\Delta F_1 = +0.5898$ (+58.98% advantage for XGBoost)
- **Bootstrap 95% Confidence Interval ($B=1,000$):**
  $$\Delta F_1 \in [0.5859, 0.5934] \quad (\text{SE} = 0.0019)$$
- **McNemar Discordance Analysis:**
  - $b$ (XGBoost correct, VAE wrong): $63,421$
  - $c$ (XGBoost wrong, VAE correct): $38$
  - Concordance ($a+d$): $61,629$ flows
  - $\chi^2 = 63,305.09 \quad (df = 1, p < 0.001)$
- **Conclusion:** XGBoost demonstrates statistically decisive superiority on attack classes represented during training.

---

### Experiment B: Held-Out Zero-Day Benchmark ($N = 172,209$)
- **Compared Metric:** Attack Recall / Detection Rate
  - Point Estimates: $\text{XGBoost Recall} = 0.6250$ | $\text{VAE Recall} = 0.1660$
  - Effect Size: $\Delta \text{Recall} = +0.4591$ (+45.91% aggregate advantage for XGBoost)
- **Bootstrap 95% Confidence Interval ($B=1,000$):**
  $$\Delta \text{Recall} \in [0.4562, 0.4619] \quad (\text{SE} = 0.0014)$$
- **McNemar Discordance Analysis:**
  - $b$ (XGBoost correct, VAE wrong): $61,482$
  - $c$ (XGBoost wrong, VAE correct): $455$
  - Concordance ($a+d$): $110,272$ flows
  - $\chi^2 = 60,128.40 \quad (df = 1, p < 0.001)$
- **Critical Qualitative Nuance:**
  - While XGBoost achieved higher aggregate recall due to volumetric DDoS overlap ($128,027$ flows), the **$c = 455$ discordance cells** where VAE was correct and XGBoost failed include **$420$ unseen attacks** (specifically Infiltration flows, where XGBoost recall was $0.0\%$ and VAE recall was $47.22\%$).

---

## 4. Correction Record in Results JSON

In [results/master_experiment_results.json](file:///c:/Users/Santhoskrishna/Documents/Cognisec_AI/results/master_experiment_results.json):
- Replaced ambiguous `"p_value_approx": 0.0` with `"p_value_bound": "< 0.001"`.
- Annotated `"metric_name": "f1_score"` for Experiment A and `"metric_name": "recall"` for Experiment B.
- Added McNemar test summary statistics ($\chi^2$, discordant cell counts $b$ and $c$).
