import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import f1_score, precision_recall_curve
from src.utils.device import get_xgboost_device

class XGBoostClassifierWrapper:
    """
    Supervised XGBoost binary classifier with validation-based threshold selection,
    feature schema auditing, and inference profiling.
    """
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        random_seed: int = 42,
        device: Optional[str] = None
    ):
        self.random_seed = random_seed
        self.device = get_xgboost_device(device)
        self.params = params or {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "device": self.device,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": self.random_seed,
            "n_jobs": -1
        }
        if "device" not in self.params:
            self.params["device"] = self.device

        try:
            self.model = xgb.XGBClassifier(**self.params)
        except Exception:
            # Fallback to CPU if params rejected device
            self.device = "cpu"
            self.params["device"] = "cpu"
            self.model = xgb.XGBClassifier(**self.params)

        self.threshold: float = 0.50
        self.feature_names: List[str] = []
        self.training_duration_seconds: float = 0.0
        self.is_trained: bool = False

    def to(self, device: str) -> "XGBoostClassifierWrapper":
        """Relocate the model's booster to the specified device (cuda or cpu)."""
        target = get_xgboost_device(device)
        self.device = target
        try:
            self.model.set_params(device=target)
            if hasattr(self.model, "get_booster"):
                self.model.get_booster().set_param({"device": target})
        except Exception:
            self.device = "cpu"
            self.model.set_params(device="cpu")
            try:
                if hasattr(self.model, "get_booster"):
                    self.model.get_booster().set_param({"device": "cpu"})
            except Exception:
                pass
        return self

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        early_stopping_rounds: int = 15,
        optimize_threshold: bool = True
    ) -> Dict[str, Any]:
        """
        Trains the XGBoost classifier with early stopping on validation split.
        Selects optimal decision threshold on validation set (if enabled).
        """
        if feature_names:
            self.feature_names = list(feature_names)

        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None

        start_time = time.time()
        if eval_set:
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)
        self.training_duration_seconds = time.time() - start_time
        self.is_trained = True

        # Optimize classification threshold strictly on validation set
        threshold_info = {"selected_threshold": self.threshold, "val_best_f1": None}
        if optimize_threshold and X_val is not None and y_val is not None:
            val_probs = self.predict_proba(X_val)
            best_thresh = 0.50
            best_f1 = 0.0

            # Scan thresholds
            for candidate in np.linspace(0.1, 0.9, 81):
                preds = (val_probs >= candidate).astype(int)
                score = f1_score(y_val, preds, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_thresh = float(candidate)

            self.threshold = best_thresh
            threshold_info["selected_threshold"] = best_thresh
            threshold_info["val_best_f1"] = float(best_f1)

        summary = {
            "training_duration_seconds": round(self.training_duration_seconds, 3),
            "num_features": X_train.shape[1],
            "train_samples": len(X_train),
            "threshold_selection": threshold_info
        }
        return summary

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> np.ndarray:
        """
        Outputs attack probability for class 1.
        Executes on configured device with automatic CPU fallback on runtime issues.
        Ensures consistent input and booster device handling without DMatrix fallback warnings.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict_proba.")

        # Standardize input to 2D float32 numpy array
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        elif isinstance(X, torch.Tensor):
            X_arr = X.detach().cpu().numpy().astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        if getattr(self, "device", "cpu") == "cuda":
            try:
                dmat = xgb.DMatrix(X_arr)
                probs = self.model.get_booster().predict(dmat)
                return probs
            except Exception:
                # Graceful CPU fallback
                self.device = "cpu"
                try:
                    self.model.set_params(device="cpu")
                    if hasattr(self.model, "get_booster"):
                        self.model.get_booster().set_param({"device": "cpu"})
                except Exception:
                    pass
                probs = self.model.predict_proba(X_arr)
                return probs[:, 1]
        else:
            probs = self.model.predict_proba(X_arr)
            return probs[:, 1]

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Outputs binary predictions using the optimized or specified threshold.
        """
        thresh = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def save(self, filepath: str) -> None:
        """
        Saves XGBoost model and threshold metadata.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save native XGBoost json
        self.model.save_model(str(path))
        # Save metadata sidecar
        meta_path = path.with_suffix(".meta.json")
        meta = {
            "threshold": self.threshold,
            "feature_names": self.feature_names,
            "training_duration_seconds": self.training_duration_seconds,
            "random_seed": self.random_seed
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)

    def load(self, filepath: str, device: Optional[str] = None) -> "XGBoostClassifierWrapper":
        """
        Loads saved model and metadata, configuring target execution device.
        """
        path = Path(filepath)
        self.model.load_model(str(path))
        target_dev = get_xgboost_device(device)
        self.device = target_dev
        try:
            self.model.set_params(device=target_dev)
            if hasattr(self.model, "get_booster"):
                self.model.get_booster().set_param({"device": target_dev})
        except Exception:
            self.device = "cpu"
            self.model.set_params(device="cpu")
            try:
                if hasattr(self.model, "get_booster"):
                    self.model.get_booster().set_param({"device": "cpu"})
            except Exception:
                pass

        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.threshold = meta.get("threshold", 0.50)
            self.feature_names = meta.get("feature_names", [])
            self.training_duration_seconds = meta.get("training_duration_seconds", 0.0)
            self.random_seed = meta.get("random_seed", 42)
        self.is_trained = True
        return self
