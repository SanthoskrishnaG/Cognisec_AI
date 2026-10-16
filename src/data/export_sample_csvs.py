"""Utility to export curated, authentic CICIDS2017 flow CSV samples for testing the Cognisec AI dashboard.
Generates:
  - data/samples/sample_benign_flows.csv (50 Benign flows)
  - data/samples/sample_known_attacks.csv (50 Known Attack flows: DoS, PortScan, Brute Force)
  - data/samples/sample_unseen_attacks.csv (50 Unseen Zero-Day flows: DDoS, Web Attack, Bot, Infiltration)
  - data/samples/sample_mixed_benchmark.csv (100 Mixed flows across all categories)
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
DATA_DIR = ROOT_DIR / "data" / "processed"
SAMPLES_DIR = ROOT_DIR / "data" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("[*] Loading test data archive and preprocessor...")
    npz_data = np.load(DATA_DIR / "test_data.npz", allow_pickle=True)
    prep = joblib.load(ROOT_DIR / "artifacts" / "models" / "flow_preprocessor.joblib")
    feature_cols = prep.feature_columns

    X_unscaled = npz_data["X_unscaled"]
    y = npz_data["y"]
    raw_labels = npz_data["raw_labels"]
    attack_families = npz_data["attack_families"]

    df_full = pd.DataFrame(X_unscaled, columns=feature_cols)
    df_full["Label"] = raw_labels
    df_full["Attack_Family"] = attack_families

    # 1. Benign Sample (50 rows)
    benign_mask = df_full["Attack_Family"] == "BENIGN"
    df_benign = df_full[benign_mask].sample(n=50, random_state=42)
    df_benign.drop(columns=["Attack_Family"]).to_csv(SAMPLES_DIR / "sample_benign_flows.csv", index=False)
    print(f"    Saved: {SAMPLES_DIR / 'sample_benign_flows.csv'} (50 rows)")

    # 2. Known Attacks Sample (50 rows: DoS, PortScan, Brute_Force)
    df_known_parts = []
    for fam in ["DoS", "PortScan", "Brute_Force"]:
        sub = df_full[df_full["Attack_Family"] == fam]
        n_take = min(len(sub), 20)
        df_known_parts.append(sub.sample(n=n_take, random_state=42))
    df_known = pd.concat(df_known_parts, ignore_index=True).head(50)
    df_known.drop(columns=["Attack_Family", "Label"]).to_csv(SAMPLES_DIR / "sample_known_attacks.csv", index=False)
    print(f"    Saved: {SAMPLES_DIR / 'sample_known_attacks.csv'} ({len(df_known)} rows)")

    # 3. Unseen / Zero-Day Attacks Sample (50 rows: DDoS, Web_Attack, Bot, Infiltration)
    df_unseen_parts = []
    for fam in ["DDoS", "Web_Attack", "Bot", "Infiltration"]:
        sub = df_full[df_full["Attack_Family"] == fam]
        n_take = min(len(sub), 15)
        df_unseen_parts.append(sub.sample(n=n_take, random_state=42))
    df_unseen = pd.concat(df_unseen_parts, ignore_index=True).head(50)
    df_unseen.drop(columns=["Attack_Family", "Label"]).to_csv(SAMPLES_DIR / "sample_unseen_attacks.csv", index=False)
    print(f"    Saved: {SAMPLES_DIR / 'sample_unseen_attacks.csv'} ({len(df_unseen)} rows)")

    # 4. Mixed Benchmark Sample (100 rows)
    df_mixed = pd.concat([
        df_benign.head(40),
        df_known.head(30),
        df_unseen.head(30)
    ], ignore_index=True)
    df_mixed.drop(columns=["Attack_Family", "Label"], errors="ignore").to_csv(SAMPLES_DIR / "sample_mixed_benchmark.csv", index=False)
    print(f"    Saved: {SAMPLES_DIR / 'sample_mixed_benchmark.csv'} ({len(df_mixed)} rows)")

    print("[SUCCESS] All sample CSV datasets successfully exported.")

if __name__ == "__main__":
    main()
