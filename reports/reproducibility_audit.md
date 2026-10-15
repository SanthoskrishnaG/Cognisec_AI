# Cognisec AI — Comprehensive Reproducibility Audit Report

**Project Title:** Cognisec AI: Network Behaviour Fingerprinting for Unseen Attack Detection — An Experimental Comparison of XGBoost and Variational Autoencoders  
**Audit Date:** 2026-10-16  
**Auditor:** Senior Research Auditor & ML Systems Engineer  
**Audit Purpose:** Formal Certification of Code, Model, Data, and Metric Determinism  

---

## 1. System & Runtime Environment Specifications

- **Operating System:** Windows 10/11 Pro (x86_64 / AMD64)
- **Host Python Executable:** `C:\Users\Santhoskrishna\Documents\Cognisec_AI\.venv\Scripts\python.exe`
- **Python Version:** `3.12.10 (tags/v3.12.10:00e708f, Feb 4 2025, 14:40:53) [MSC v.1942 64 bit (AMD64)]`
- **Global Random Seed:** `42` (Enforced across NumPy, PyTorch CPU/CUDA, Python `random`, and XGBoost `random_state`)

### Installed Package Manifest
- `torch`: `2.14.1+cpu`
- `xgboost`: `3.4.1`
- `scikit-learn`: `1.9.1`
- `pandas`: `3.0.6`
- `numpy`: `2.2.6`
- `scipy`: `1.16.1`
- `flask`: `3.1.3`
- `joblib`: `1.5.1`
- `pytest`: `9.1.1`
- `requests`: `2.32.3`
- `streamlit`: `1.66.0`

---

## 2. Dataset Files & Cryptographic Hashes (SHA-256)

### Raw CICIDS2017 Benchmark Files (`data/raw/`)
| Filename | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :--- |
| `Friday-WorkingHours-Afternoon-DDos.csv` | 57,055,277 | `975dfab7dd6447977c2b9474afc872cfcb6dc83a87603393290c5046b79995e2` |
| `Friday-WorkingHours-Afternoon-PortScan.csv` | 52,205,654 | `08dbc58a8551e8d94826c7a9acf328828aa5046c5cb6afc702fa56c97fcdfb21` |
| `Friday-WorkingHours-Morning.csv` | 9,014,478 | `97861740269db3e02d431c7dd3d29269cd1212c84c79b5280ade0668c2acf15a` |
| `Monday-WorkingHours.csv` | 9,029,716 | `3f9f2ad217dbbc6fa3bb25070312c6d20512da48af30d9003dbe4d58dc017862` |
| `Thursday-WorkingHours-Afternoon-Infilteration.csv` | 7,979,975 | `63ae73dc781daa2f7102f1d3517ea7b5d1b60c420e6a435d6df75df0aa9bd857` |
| `Thursday-WorkingHours-Morning-WebAttacks.csv` | 9,091,484 | `ab94606ff45104286a67855f721250fde2a8bfd1cd1de8b3ad9efd512bdde98d` |
| `Tuesday-WorkingHours.csv` | 13,229,339 | `6644b8bf2a5d713d37948350959f2042f055fc90f53e183da2dab90e8fc35867` |
| `Wednesday-workingHours.csv` | 105,302,558 | `c9eb37024ff893abfa4d5aaeeb7b261a7f1a8c22dbf1cdfb251ea637b470382f` |

### Partitioned Data Arrays (`data/processed/`)
| Filename | Records | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| `train_data.npz` | 375,262 | 54,846,232 | `413261921f60f56d577b10a82e0f50fedcc6a6c40cacf76d1979809f13435dd9` |
| `val_data.npz` | 125,087 | 18,374,129 | `acf3821f0f41afdf974395ff7c53ef410ea92a3278453a46767ab319c216bb63` |
| `test_data.npz` | 257,297 | 38,594,631 | `1df3ac5a8fd103a01db186842d2df491bb3b85920306d0cc927e9e319f4f8128` |
| `partition_manifest.json` | Manifest | 1,251 | `7d33d30d72b51a03799139f3fa58d9a49c1abbea48bc5a95c182b6643f915fde` |

---

## 3. Model Artifacts & Cryptographic Signatures

| Artifact File | Description | Parameters / Threshold | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| `artifacts/models/flow_preprocessor.joblib` | Fitted `FlowPreprocessor` (70 clean features) | Median imputer + log1p + `RobustScaler` | `a9bc7f32ce1806a10081aa1d755a7872d8df528a32d04d89fdc42b24d82e22cf` |
| `artifacts/models/xgboost_model.json` | Supervised Histogram XGBoost | `n_est=200`, `depth=6`, $\tau = 0.5200$ | `5f6e61c8dde5aaffd88fd4471eba744a94307fb58e5d103fbed887ca1264fe52` |
| `artifacts/models/xgboost_model.meta.json` | XGBoost sidecar metadata | 70 feature names, threshold 0.52 | `4ab4fa021b49b17ec9b107365455b777f6b6267133a84eda7ec7fcd7b4f390c8` |
| `artifacts/models/vae_checkpoint.pt` | PyTorch Tabular VAE State Dict | $70 \to [64,32] \to 16 \to [32,64] \to 70$, $\tau = 0.330610$ | `3e824e46a29b739eeee3dea8ad08fff24c27534bf53def02ed0de7cf8820c0e7` |

---

## 4. Controlled Reproducibility Trial Results

To test bit-for-bit repeatability, fresh models were trained from scratch using the exact configuration files (`configs/xgboost.yaml` and `configs/vae.yaml`) into temporary scratch files and evaluated against the identical test partition:

```
[+] Controlled XGBoost Training: Completed in 4.78s. Validation threshold: 0.5200 (100% Match)
[+] Controlled VAE Training:     Completed in 21.83s. Validation threshold: 0.330610 (100% Match)
```

### Direct Metric Comparison (Original vs. Fresh Retrain)
- **XGBoost Known Attack F1:** Original = `0.9997238656459487` | Fresh = `0.9997238656459487` ($\Delta = 0.0$)
- **XGBoost Held-Out Recall:** Original = `0.6250330915444485` | Fresh = `0.6250330915444485` ($\Delta = 0.0$)
- **VAE Known Attack F1:** Original = `0.40989859521815986` | Fresh = `0.40989859521815986` ($\Delta = 0.0$)
- **VAE Held-Out Recall:** Original = `0.16595693182763654` | Fresh = `0.16595693182763654` ($\Delta = 0.0$)
- **Infiltration Zero-Day Recall:** Original = `47.22%` | Fresh = `47.22%` ($\Delta = 0.0$)

**Audit Verdict:** Both model training algorithms produce deterministic outputs when anchored to `random_seed=42`.

---

## 5. Potential Sources of Non-Determinism & Safeguards

| Source of Non-Determinism | Assessment | Implemented Safeguard |
| :--- | :--- | :--- |
| **Multi-threaded CPU execution** | OpenMP threads in XGBoost or PyTorch can introduce minor floating-point sum reordering. | Fixed `random_state=42` and single-process inference loops; tolerance set to $\le 10^{-6}$. |
| **GPU Acceleration Non-Determinism** | Atomic additions on CUDA devices can cause minor gradient drift. | Production models and benchmarks are executed deterministically on CPU (`device="cpu"`). |
| **Python Dictionary Ordering** | Hash seed randomization in Python versions $\ge 3.8$. | Feature column ordering is strictly enforced by `preprocessor.feature_columns` list. |
| **OS Filesystem Path Casing** | Windows case-insensitivity causing virtual environment re-launch loops. | Enforced filesystem identity check using `os.path.samefile()`. |
