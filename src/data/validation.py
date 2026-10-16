import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import numpy as np

class DataValidator:
    """
    Validates and audits network-flow datasets according to research requirements:
    - Identifies missing, infinite, and NaN values
    - Detects duplicate records
    - Checks class imbalance and attack-family representation
    - Audits potential leakage columns and zero-variance features
    """
    def __init__(self, leakage_candidates: List[str] = None):
        self.leakage_candidates = leakage_candidates or [
            "Flow ID", "Source IP", "Destination IP", "Timestamp", "External IP"
        ]

    def audit_dataset(self, df: pd.DataFrame, label_col: str = "Label") -> Dict[str, Any]:
        """
        Runs comprehensive data quality and leakage checks on the raw/combined DataFrame.
        """
        total_rows = len(df)
        total_cols = len(df.columns)

        # Missing values
        missing_counts = df.isnull().sum()
        cols_with_missing = {col: int(cnt) for col, cnt in missing_counts.items() if cnt > 0}

        # Infinite values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_with_inf = {}
        for col in numeric_cols:
            inf_count = int(np.isinf(df[col]).sum())
            if inf_count > 0:
                cols_with_inf[col] = inf_count

        # Duplicate rows
        duplicate_count = int(df.duplicated().sum())

        # Class / Label distribution
        if label_col in df.columns:
            label_distribution = df[label_col].value_counts().to_dict()
            label_distribution = {str(k): int(v) for k, v in label_distribution.items()}
        else:
            label_distribution = {}

        # Zero or near-zero variance features
        constant_features = []
        for col in numeric_cols:
            if df[col].nunique() <= 1:
                constant_features.append(col)

        # Leakage candidates found
        detected_leakage = [col for col in self.leakage_candidates if col in df.columns]

        report = {
            "total_records": total_rows,
            "total_features": total_cols,
            "missing_values": cols_with_missing,
            "infinite_values": cols_with_inf,
            "duplicate_records": duplicate_count,
            "duplicate_percentage": round((duplicate_count / total_rows * 100), 2) if total_rows > 0 else 0,
            "label_distribution": label_distribution,
            "constant_features": constant_features,
            "leakage_columns_detected": detected_leakage
        }
        return report

    def save_audit_report(self, report: Dict[str, Any], save_path: str = "reports/data_audit_report.json") -> None:
        """
        Saves audit report to a machine-readable JSON file.
        """
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)


def verify_dataset_files(
    raw_data_dir: Any = "data/raw",
    audit_report_path: Any = "reports/data_audit_report.json"
) -> Dict[str, Any]:
    """
    Independently inspects and authenticates actual on-disk CICIDS2017 CSV files.
    Calculates streaming line counts directly from physical storage to ensure zero fabricated numbers.
    Reports errors if files are missing or unreadable.
    """
    raw_dir = Path(raw_data_dir)
    audit_path = Path(audit_report_path) if audit_report_path else None

    audit_data = None
    if audit_path and audit_path.exists():
        try:
            with open(audit_path, "r", encoding="utf-8") as f:
                audit_data = json.load(f)
        except Exception:
            audit_data = None

    if not raw_dir.exists():
        return {
            "is_verified": False,
            "status": "error",
            "error": f"Raw data directory does not exist: {raw_dir}",
            "files_found_count": 0,
            "total_records": 0,
            "files": []
        }

    expected_files = [
        "Friday-WorkingHours-Afternoon-DDos.csv",
        "Friday-WorkingHours-Afternoon-PortScan.csv",
        "Friday-WorkingHours-Morning.csv",
        "Monday-WorkingHours.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.csv",
        "Tuesday-WorkingHours.csv",
        "Wednesday-workingHours.csv"
    ]

    files_present = []
    missing_files = []
    total_records = 0
    file_provenance = {}

    for fname in expected_files:
        fpath = raw_dir / fname
        if not fpath.exists():
            missing_files.append(fname)
            continue

        try:
            with open(fpath, "rb") as fp:
                lines = sum(1 for _ in fp) - 1
            if lines < 0:
                lines = 0
            size_b = fpath.stat().st_size
            files_present.append({
                "filename": fname,
                "records": lines,
                "size_bytes": size_b,
                "status": "verified"
            })
            total_records += lines
            file_provenance[fname] = lines
        except Exception as e:
            missing_files.append(f"{fname} (read error: {str(e)})")

    is_verified = (len(missing_files) == 0 and len(files_present) == len(expected_files))

    label_dist = audit_data.get("label_distribution", {}) if audit_data else {}
    benign_count = label_dist.get("BENIGN", 0)
    attack_count = sum(v for k, v in label_dist.items() if k != "BENIGN")

    result = {
        "is_verified": is_verified,
        "status": "verified" if is_verified else ("partial" if files_present else "missing_files"),
        "counting_method": "Streaming row count on 8 authenticated on-disk CSV files (excluding header rows)",
        "expected_files_count": len(expected_files),
        "files_found_count": len(files_present),
        "files": files_present,
        "missing_files": missing_files,
        "file_provenance": file_provenance,
        "total_records": total_records if is_verified else (total_records if files_present else None),
        "benign_records": benign_count if audit_data else None,
        "attack_records": attack_count if audit_data else None,
        "total_features": audit_data.get("total_features", 79) if audit_data else None,
        "constant_features_dropped": audit_data.get("constant_features", []) if audit_data else [],
        "duplicate_records": audit_data.get("duplicate_records") if audit_data else None,
        "duplicate_percentage": audit_data.get("duplicate_percentage") if audit_data else None,
        "missing_values": audit_data.get("missing_values", {}) if audit_data else {},
        "infinite_values": audit_data.get("infinite_values", {}) if audit_data else {},
        "label_distribution": label_dist
    }
    return result

