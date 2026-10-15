import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.ingestion import DatasetIngestion
from src.data.validation import DataValidator
from src.data.preprocessing import FlowPreprocessor
from src.data.splitting import DataSplitter
from src.utils.configuration import load_yaml_config
from src.utils.reproducibility import set_seed

def run_data_preparation_pipeline():
    print("[*] Starting Phase 3: Dataset preprocessing and leakage-controlled partitioning...")
    set_seed(42)
    cfg = load_yaml_config("configs/data.yaml")

    # 1. Ingestion
    ingestion = DatasetIngestion(raw_data_dir=cfg.get("raw_data_dir", "data/raw"))
    combined_df, provenance = ingestion.load_all_files(max_rows_per_file=None)
    print(f"[+] Loaded {len(combined_df)} records across {len(provenance)} authentic files.")

    # 2. Validation & Audit
    validator = DataValidator(leakage_candidates=cfg.get("leakage_columns", []))
    audit_report = validator.audit_dataset(combined_df, label_col=cfg.get("label_column", "Label"))
    audit_report["file_provenance"] = provenance
    validator.save_audit_report(audit_report, save_path="reports/data_audit_report.json")
    print(f"[+] Data audit complete. Found {len(audit_report['label_distribution'])} raw label variants.")

    # 3. Map to standard Attack Families
    family_mapping = cfg.get("attack_family_mapping", {})
    label_col = cfg.get("label_column", "Label")
    
    def map_family(val):
        clean_val = str(val).strip()
        for k, v in family_mapping.items():
            if clean_val.lower() == k.lower():
                return v
        if "web attack" in clean_val.lower():
            return "Web_Attack"
        if "dos" in clean_val.lower():
            return "DoS"
        if "patator" in clean_val.lower():
            return "Brute_Force"
        return clean_val

    combined_df["Attack_Family"] = combined_df[label_col].apply(map_family)
    print("[+] Attack Family Distribution:")
    for fam, cnt in combined_df["Attack_Family"].value_counts().items():
        print(f"    - {fam}: {cnt:,} records")

    # 4. Leakage-Controlled Partitioning
    held_out_families = cfg.get("held_out_attack_families", ["DDoS", "Web_Attack", "Bot", "Infiltration"])
    splitter = DataSplitter(
        label_col=label_col,
        benign_label=cfg.get("benign_label", "BENIGN"),
        train_ratio=cfg.get("train_ratio", 0.60),
        val_ratio=cfg.get("val_ratio", 0.20),
        test_ratio=cfg.get("test_ratio", 0.20),
        random_seed=cfg.get("random_seed", 42)
    )

    train_df, val_df, test_df, manifest = splitter.create_partitions(
        df=combined_df,
        held_out_families=held_out_families,
        family_col="Attack_Family"
    )
    splitter.save_manifest(manifest, filepath=cfg.get("manifest_path", "data/processed/partition_manifest.json"))
    print(f"[+] Partitions created: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
    print(f"    -> Held-out families strictly in Test: {held_out_families}")

    # 5. Preprocessing & Scaling (Fitted exclusively on Training Data)
    preprocessor = FlowPreprocessor(
        leakage_columns=cfg.get("leakage_columns", []),
        label_column=label_col,
        binary_target_column=cfg.get("binary_target_column", "Target"),
        benign_label=cfg.get("benign_label", "BENIGN"),
        scaler_type="robust"
    )

    print("[*] Fitting FlowPreprocessor strictly on Train partition...")
    preprocessor.fit(train_df)
    preprocessor_path = "artifacts/models/flow_preprocessor.joblib"
    preprocessor.save(preprocessor_path)
    print(f"[+] Preprocessor fitted and saved. Retained features: {len(preprocessor.feature_columns)}")

    # 6. Transform partitions
    X_train_unscaled, X_train_scaled, y_train, raw_labels_train = preprocessor.transform(train_df)
    X_val_unscaled, X_val_scaled, y_val, raw_labels_val = preprocessor.transform(val_df)
    X_test_unscaled, X_test_scaled, y_test, raw_labels_test = preprocessor.transform(test_df)

    # Save processed arrays in data/processed/
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        processed_dir / "train_data.npz",
        X_unscaled=X_train_unscaled,
        X_scaled=X_train_scaled,
        y=y_train,
        raw_labels=raw_labels_train,
        attack_families=train_df["Attack_Family"].values.astype(str)
    )
    np.savez_compressed(
        processed_dir / "val_data.npz",
        X_unscaled=X_val_unscaled,
        X_scaled=X_val_scaled,
        y=y_val,
        raw_labels=raw_labels_val,
        attack_families=val_df["Attack_Family"].values.astype(str)
    )
    np.savez_compressed(
        processed_dir / "test_data.npz",
        X_unscaled=X_test_unscaled,
        X_scaled=X_test_scaled,
        y=y_test,
        raw_labels=raw_labels_test,
        attack_families=test_df["Attack_Family"].values.astype(str)
    )

    print("[+] Processed numpy arrays saved to data/processed/")
    print(f"    Train shape: {X_train_unscaled.shape}, Class balance: {np.bincount(y_train).tolist()}")
    print(f"    Val shape:   {X_val_unscaled.shape}, Class balance: {np.bincount(y_val).tolist()}")
    print(f"    Test shape:  {X_test_unscaled.shape}, Class balance: {np.bincount(y_test).tolist()}")
    print("[+] Phase 3 successfully completed.")

if __name__ == "__main__":
    run_data_preparation_pipeline()
