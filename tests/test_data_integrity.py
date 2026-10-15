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


def test_record_count_and_confusion_matrix_invariants():
    """
    Validates Task 2 invariants:
    1. 257,297 unique physical records in test partition.
    2. Exactly 297,297 total benchmark evaluations across Exp A & Exp B.
    3. Exactly 40,000 shared benign control flows.
    4. Exact confusion matrix sums for both models and both experiments.
    5. Complementarity sum equals total held-out attacks (132,209).
    """
    master_path = Path("results/master_experiment_results.json")
    assert master_path.exists(), "master_experiment_results.json not found"

    with open(master_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Total record reconciliation invariants
    assert data["total_unique_test_records"] == 257297
    assert data["total_benchmark_evaluations"] == 297297
    assert data["shared_benign_control_records"] == 40000

    exp_a = data["experiment_a_known"]
    exp_b = data["experiment_b_held_out"]

    # 2. Experiment A sums
    xgb_a = exp_a["xgboost"]
    vae_a = exp_a["vae"]
    total_a = xgb_a["true_negatives"] + xgb_a["false_positives"] + xgb_a["false_negatives"] + xgb_a["true_positives"]
    total_vae_a = vae_a["true_negatives"] + vae_a["false_positives"] + vae_a["false_negatives"] + vae_a["true_positives"]
    assert total_a == 125088
    assert total_vae_a == 125088
    assert xgb_a["true_negatives"] + xgb_a["false_positives"] == 40000  # Benign (39961 + 39)
    assert xgb_a["false_negatives"] + xgb_a["true_positives"] == 85088  # Known attacks (85080 + 8)
    assert vae_a["false_negatives"] + vae_a["true_positives"] == 85088  # Known attacks (22030 + 63058)


    # 3. Experiment B sums
    xgb_b = exp_b["xgboost_summary"]
    vae_b = exp_b["vae_summary"]
    total_b = xgb_b["true_negatives"] + xgb_b["false_positives"] + xgb_b["false_negatives"] + xgb_b["true_positives"]
    total_vae_b = vae_b["true_negatives"] + vae_b["false_positives"] + vae_b["false_negatives"] + vae_b["true_positives"]
    assert total_b == 172209
    assert total_vae_b == 172209
    assert xgb_b["true_negatives"] + xgb_b["false_positives"] == 40000  # Benign control
    assert xgb_b["false_negatives"] + xgb_b["true_positives"] == 132209 # Held-out attacks

    # 4. Physical uniqueness invariant
    assert total_a + total_b - 40000 == 257297

    # 5. Complementarity partition
    comp = data["experiment_f_complementarity"]
    comp_sum = comp["both_detected"] + comp["xgboost_only"] + comp["vae_only"] + comp["both_missed"]
    assert comp_sum == 132209



def test_statistical_validation_invariants():
    """
    Validates Task 3 invariants:
    1. Statistical comparison bounds exist and do not report invalid p = 0.0.
    2. Bootstrap 95% confidence intervals are mathematically ordered (lower < point < upper).
    3. McNemar's test results are present with valid chi-squared statistics.
    """
    master_path = Path("results/master_experiment_results.json")
    with open(master_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for exp_key in ["experiment_a_known", "experiment_b_held_out"]:
        stat = data[exp_key]["paired_comparison"]
        assert "p_value_bound" in stat
        assert stat["p_value_bound"] in ["< 0.001", "< 1e-6"]
        ci_lower = stat["ci_lower"]
        ci_upper = stat["ci_upper"]
        point = stat["delta_point"]
        assert ci_lower <= point <= ci_upper
        assert "mcnemar_test" in stat
        assert stat["mcnemar_test"]["chi2_statistic"] > 1000.0


