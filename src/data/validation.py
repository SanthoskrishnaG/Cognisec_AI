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
