# Master Experimental Comparison Table

| Metric | XGBoost (Supervised) | VAE (Benign Anomaly) | Difference (Δ) | Statistical Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Known Attack PR-AUC** | 1.0000 | 0.9432 | 0.0568 | Baseline Comparison |
| **Known Attack F1-Score** | 0.9997 | 0.4099 | 0.5898 | 95% CI: [0.5859, 0.5938], p=0.0 |
| **Known Attack Recall** | 0.9999 | 0.2589 | 0.7410 | -- |
| **Known Attack FPR** | 0.0010 | 0.0093 | -0.0083 | Target: 1.0% |
| **Held-Out Family PR-AUC** | 0.9924 | 0.9802 | 0.0122 | Zero-Day Benchmark |
| **Held-Out Family Recall** | 0.6250 | 0.1660 | 0.4591 | 95% CI: [0.4568, 0.4619], p=0.0 |
| **Inference Latency** | 0.0005 ms/flow | 0.0052 ms/flow | -- | CPU Throughput Benchmark |
