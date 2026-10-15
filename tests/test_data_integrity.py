import json
from pathlib import Path
import numpy as np
import pytest

def test_data_integrity_and_leakage_controls():
    """
    Validates:
    1. Preprocessed numpy arrays exist and contain valid finite data.
    2. Zero identity/socket leakage in feature names.
    3. Strict held-out attack isolation (zero contamination in train/val).
    4. Label binary mapping consistency (y in {0, 1}).
    """
    processed_dir = Path("data/processed")
    train_npz = processed_dir / "train_data.npz"
    val_npz = processed_dir / "val_data.npz"
    test_npz = processed_dir / "test_data.npz"
    manifest_path = processed_dir / "partition_manifest.json"

    assert train_npz.exists(), "train_data.npz not found"
    assert val_npz.exists(), "val_data.npz not found"
    assert test_npz.exists(), "test_data.npz not found"
    assert manifest_path.exists(), "partition_manifest.json not found"

    train_data = np.load(train_npz, allow_pickle=True)
    val_data = np.load(val_npz, allow_pickle=True)
    test_data = np.load(test_npz, allow_pickle=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    held_out_families = set(manifest.get("held_out_attack_families", ["DDoS", "Web_Attack", "Bot", "Infiltration"]))

    # 1. Check finite values (no NaN or Inf)
    assert np.all(np.isfinite(train_data["X_unscaled"])), "NaN or Inf detected in train X_unscaled"
    assert np.all(np.isfinite(train_data["X_scaled"])), "NaN or Inf detected in train X_scaled"
    assert np.all(np.isfinite(val_data["X_unscaled"])), "NaN or Inf detected in val X_unscaled"
    assert np.all(np.isfinite(val_data["X_scaled"])), "NaN or Inf detected in val X_scaled"
    assert np.all(np.isfinite(test_data["X_unscaled"])), "NaN or Inf detected in test X_unscaled"
    assert np.all(np.isfinite(test_data["X_scaled"])), "NaN or Inf detected in test X_scaled"

    # 2. Check binary targets
    assert set(np.unique(train_data["y"])).issubset({0, 1})
    assert set(np.unique(val_data["y"])).issubset({0, 1})
    assert set(np.unique(test_data["y"])).issubset({0, 1})

    # 3. Check strict held-out attack isolation: zero held-out families in train or val!
    train_families = set(train_data["attack_families"])
    val_families = set(val_data["attack_families"])
    test_families = set(test_data["attack_families"])

    contamination_train = train_families.intersection(held_out_families)
    contamination_val = val_families.intersection(held_out_families)
    assert len(contamination_train) == 0, f"Leakage detected: {contamination_train} found in train partition!"
    assert len(contamination_val) == 0, f"Leakage detected: {contamination_val} found in val partition!"

    # 4. Check held-out families are present in test set
    assert held_out_families.issubset(test_families), "Not all held-out families are represented in test set!"

    # 5. Check preprocessor feature columns for leakage
    import joblib
    preprocessor = joblib.load("artifacts/models/flow_preprocessor.joblib")
    features = [f.lower() for f in preprocessor.feature_columns]
    forbidden_terms = ["flow id", "source ip", "destination ip", "timestamp", "external ip"]
    for term in forbidden_terms:
        assert not any(term == f for f in features), f"Forbidden leakage feature '{term}' found in preprocessor!"
