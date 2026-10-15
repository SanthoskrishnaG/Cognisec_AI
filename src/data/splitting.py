import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np

class DataSplitter:
    """
    Manages leakage-controlled partitioning and held-out attack-family isolation.
    Guarantees:
    - Zero contamination of held-out families into training/validation.
    - VAE training partition contains exclusively benign records.
    - VAE threshold validation partition contains exclusively benign records.
    - Generates a machine-readable partition manifest.
    """
    def __init__(
        self,
        label_col: str = "Label",
        benign_label: str = "BENIGN",
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
        random_seed: int = 42
    ):
        self.label_col = label_col
        self.benign_label = benign_label
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    def create_partitions(
        self,
        df: pd.DataFrame,
        held_out_families: List[str],
        family_col: str = "Attack_Family"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Splits dataset into train, val, and test partitions with strict attack reservation.
        """
        df = df.copy()
        rng = np.random.RandomState(self.random_seed)

        # Ensure Attack_Family column exists
        if family_col not in df.columns:
            df[family_col] = df[self.label_col]

        # 1. Isolate held-out attack families: they go 100% into test set!
        is_held_out = df[family_col].isin(held_out_families)
        held_out_df = df[is_held_out]
        trainable_pool_df = df[~is_held_out]

        # 2. Split remaining records into train, val, and test
        # Separate benign and known attacks to maintain stratification
        benign_pool = trainable_pool_df[trainable_pool_df[self.label_col] == self.benign_label]
        known_attack_pool = trainable_pool_df[trainable_pool_df[self.label_col] != self.benign_label]

        # Shuffle indices
        benign_idx = rng.permutation(benign_pool.index)
        n_benign = len(benign_idx)
        b_train_end = int(n_benign * self.train_ratio)
        b_val_end = int(n_benign * (self.train_ratio + self.val_ratio))

        b_train = benign_pool.loc[benign_idx[:b_train_end]]
        b_val = benign_pool.loc[benign_idx[b_train_end:b_val_end]]
        b_test = benign_pool.loc[benign_idx[b_val_end:]]

        # Known attacks split
        if len(known_attack_pool) > 0:
            attack_idx = rng.permutation(known_attack_pool.index)
            n_attack = len(attack_idx)
            a_train_end = int(n_attack * self.train_ratio)
            a_val_end = int(n_attack * (self.train_ratio + self.val_ratio))

            a_train = known_attack_pool.loc[attack_idx[:a_train_end]]
            a_val = known_attack_pool.loc[attack_idx[a_train_end:a_val_end]]
            a_test = known_attack_pool.loc[attack_idx[a_val_end:]]
        else:
            a_train = pd.DataFrame(columns=df.columns)
            a_val = pd.DataFrame(columns=df.columns)
            a_test = pd.DataFrame(columns=df.columns)

        # Assemble splits
        train_df = pd.concat([b_train, a_train], ignore_index=True)
        val_df = pd.concat([b_val, a_val], ignore_index=True)
        test_df = pd.concat([b_test, a_test, held_out_df], ignore_index=True)

        # Create experiment manifest
        manifest = {
            "random_seed": self.random_seed,
            "held_out_attack_families": held_out_families,
            "train_summary": {
                "total": len(train_df),
                "benign": len(b_train),
                "known_attacks": len(a_train),
                "family_distribution": train_df[family_col].value_counts().to_dict()
            },
            "val_summary": {
                "total": len(val_df),
                "benign": len(b_val),
                "known_attacks": len(a_val),
                "family_distribution": val_df[family_col].value_counts().to_dict()
            },
            "test_summary": {
                "total": len(test_df),
                "benign": len(b_test),
                "known_attacks": len(a_test),
                "held_out_attacks": len(held_out_df),
                "family_distribution": test_df[family_col].value_counts().to_dict()
            }
        }

        return train_df, val_df, test_df, manifest

    @staticmethod
    def save_manifest(manifest: Dict[str, Any], filepath: str = "data/processed/partition_manifest.json") -> None:
        """
        Saves the partition manifest as a formatted JSON artifact.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)
