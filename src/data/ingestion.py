import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

class DatasetIngestion:
    """
    Ingests authentic CICIDS2017 machine-learning flow CSV files.
    Standardises column headers, downcasts numeric types for memory efficiency,
    and documents data provenance.
    """
    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_files: List[str] = []

    def discover_csv_files(self) -> List[Path]:
        """
        Finds all CSV files within the raw data directory.
        """
        files = sorted(list(self.raw_data_dir.glob("*.csv")))
        return files

    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Strips extra whitespace and control characters from column names.
        """
        df.columns = [col.strip() for col in df.columns]
        return df

    def load_single_file(
        self,
        file_path: Union[str, Path],
        nrows: Optional[int] = None,
        usecols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Loads a single CICIDS2017 CSV file with robust encoding, type casting,
        and header sanitisation.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        # Attempt UTF-8, fallback to latin-1 for legacy CSVs
        try:
            df = pd.read_csv(path, nrows=nrows, usecols=usecols, low_memory=False, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, nrows=nrows, usecols=usecols, low_memory=False, encoding="latin-1")

        df = self.clean_column_names(df)

        # Optimize numeric columns to 32-bit floats
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype(np.float32)
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = df[col].astype(np.int32)

        return df

    def load_all_files(
        self,
        max_rows_per_file: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Loads and concatenates all available CICIDS2017 CSVs.
        Returns combined DataFrame and a provenance dictionary of record counts per file.
        """
        csv_files = self.discover_csv_files()
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.raw_data_dir}")

        dfs = []
        provenance = {}
        for file in csv_files:
            df_file = self.load_single_file(file, nrows=max_rows_per_file)
            provenance[file.name] = len(df_file)
            dfs.append(df_file)
            self.loaded_files.append(str(file))

        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df, provenance
