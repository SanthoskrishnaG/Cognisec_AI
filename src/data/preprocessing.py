import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

class FlowPreprocessor:
    """
    Leakage-controlled preprocessing pipeline for network flow records.
    Fits all scalers, imputers, and schemas exclusively on training data.
    Provides tailored representations for XGBoost (unscaled/imputed) and VAE (standardized/robust scaled).
    """
    def __init__(
        self,
        leakage_columns: Optional[List[str]] = None,
        label_column: str = "Label",
        binary_target_column: str = "Target",
        benign_label: str = "BENIGN",
        scaler_type: str = "robust"
    ):
        self.leakage_columns = leakage_columns or [
            "Flow ID", "Source IP", "Destination IP", "Timestamp", "External IP"
        ]
        self.label_column = label_column
        self.binary_target_column = binary_target_column
        self.benign_label = benign_label
        self.scaler_type = scaler_type

        self.feature_columns: List[str] = []
        self.median_values: Dict[str, float] = {}
        self.constant_columns_to_drop: List[str] = []
        self.scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.is_fitted = False

    def clean_raw_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strips leakage columns, creates binary target, and replaces infinities.
        """
        df = df.copy()

        # Remove known leakage columns if present
        cols_to_drop = [c for c in self.leakage_columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Standardise label string whitespace
        if self.label_column in df.columns:
            df[self.label_column] = df[self.label_column].astype(str).str.strip()
            # Binary target: 0 = Benign, 1 = Attack
            df[self.binary_target_column] = (df[self.label_column] != self.benign_label).astype(int)

        # Replace infinite values with NaN across all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != self.binary_target_column:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        return df

    def fit(self, train_df: pd.DataFrame) -> "FlowPreprocessor":
        """
        Learns imputations, identifies zero-variance features, and fits the scaler
        exclusively on the training split.
        """
        df = self.clean_raw_frame(train_df)

        exclude_cols = {self.label_column, self.binary_target_column, "Attack_Family"}
        candidate_features = [c for c in df.columns if c not in exclude_cols]

        # Identify constant columns (zero variance)
        self.constant_columns_to_drop = []
        for col in candidate_features:
            if df[col].nunique(dropna=True) <= 1:
                self.constant_columns_to_drop.append(col)

        # Retained feature columns
        self.feature_columns = [
            c for c in candidate_features if c not in self.constant_columns_to_drop
        ]

        # Calculate medians for imputation from training data only
        self.median_values = {}
        for col in self.feature_columns:
            med = float(df[col].median(skipna=True))
            if np.isnan(med):
                med = 0.0
            self.median_values[col] = med

        # Impute training features
        imputed_features = df[self.feature_columns].fillna(self.median_values).values.astype(np.float32)

        # Variance-stabilizing transformation for VAE: log1p on non-negative values
        log_imputed = np.log1p(np.maximum(0, imputed_features))

        # Fit feature scaler exclusively on training data
        self.scaler = StandardScaler()
        self.scaler.fit(log_imputed)
        self.is_fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Transforms data into model-ready arrays.
        Returns:
            X_unscaled: np.ndarray (for XGBoost)
            X_scaled: np.ndarray (log1p + standard scaled for VAE)
            y_binary: np.ndarray (0 or 1)
            raw_labels: np.ndarray (original string labels for family-wise evaluation)
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before calling transform.")

        df = self.clean_raw_frame(df)

        # Align columns and impute
        imputed_df = pd.DataFrame(index=df.index)
        for col in self.feature_columns:
            if col in df.columns:
                imputed_df[col] = df[col].fillna(self.median_values[col])
            else:
                imputed_df[col] = self.median_values[col]

        X_unscaled = imputed_df[self.feature_columns].values.astype(np.float32)
        X_log = np.log1p(np.maximum(0, X_unscaled))
        X_scaled = self.scaler.transform(X_log).astype(np.float32)
        # Bounded clipping to prevent extreme numerical divergence
        X_scaled = np.clip(X_scaled, -5.0, 5.0)

        y_binary = df[self.binary_target_column].values.astype(np.int32) if self.binary_target_column in df.columns else np.zeros(len(df), dtype=np.int32)
        raw_labels = df[self.label_column].values.astype(str) if self.label_column in df.columns else np.array(["UNKNOWN"] * len(df))

        return X_unscaled, X_scaled, y_binary, raw_labels

    def save(self, filepath: str) -> None:
        """
        Saves the fitted preprocessor pipeline to disk.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: str) -> "FlowPreprocessor":
        """
        Loads fitted preprocessor pipeline.
        """
        return joblib.load(filepath)
