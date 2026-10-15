import json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from src.models.xgboost_model import XGBoostClassifierWrapper
from src.evaluation.metrics import calculate_classification_metrics
from src.utils.configuration import load_yaml_config

def train_xgboost_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Optional[list] = None,
    config_path: str = "configs/xgboost.yaml"
) -> Dict[str, Any]:
    """
    Executes supervised XGBoost model training and validation threshold tuning.
    """
    cfg = load_yaml_config(config_path)
    params = cfg.get("params", {})
    random_seed = cfg.get("random_seed", 42)
    save_path = cfg.get("model_save_path", "artifacts/models/xgboost_model.json")
    early_stopping = cfg.get("early_stopping_rounds", 15)
    opt_threshold = cfg.get("optimize_threshold", True)

    classifier = XGBoostClassifierWrapper(params=params, random_seed=random_seed)
    train_summary = classifier.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        early_stopping_rounds=early_stopping,
        optimize_threshold=opt_threshold
    )

    # Evaluate on validation split
    val_preds = classifier.predict(X_val)
    val_probs = classifier.predict_proba(X_val)
    val_metrics = calculate_classification_metrics(y_true=y_val, y_pred=val_preds, y_score=val_probs)

    # Save trained model
    classifier.save(save_path)

    results = {
        "model_name": "XGBoost",
        "training_summary": train_summary,
        "validation_metrics": val_metrics,
        "saved_model_path": save_path
    }
    return results
