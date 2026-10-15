# Master Experimental Comparison Table

| Metric | XGBoost (Supervised) | VAE (Benign Anomaly) | Difference (Δ) | Statistical Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Known Attack PR-AUC** | 1.0000 | 0.5974 | 0.4026 | Baseline Comparison |
| **Known Attack F1-Score** | 0.9997 | 0.0183 | 0.9814 | 95% CI: [0.9801, 0.9826], p=0.0 |
| **Known Attack Recall** | 0.9999 | 0.0093 | 0.9906 | -- |
| **Known Attack FPR** | 0.0010 | 0.0117 | -0.0107 | Target: 1.0% |
| **Held-Out Family PR-AUC** | 0.9924 | 0.7077 | 0.2847 | Zero-Day Benchmark |
| **Held-Out Family Recall** | 0.6250 | 0.0909 | 0.5342 | 95% CI: [0.5319, 0.5371], p=0.0 |
| **Inference Latency** | 0.0005 ms/flow | 0.0059 ms/flow | -- | CPU Throughput Benchmark |
