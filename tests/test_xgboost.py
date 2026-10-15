import numpy as np
import pytest
from src.models.xgboost_model import XGBoostClassifierWrapper

def test_xgboost_training_and_inference(tmp_path):
    np.random.seed(42)
    # Synthetic mini-batch for unit test
    X_train = np.random.randn(100, 10).astype(np.float32)
    y_train = (X_train[:, 0] > 0).astype(int)
    X_val = np.random.randn(40, 10).astype(np.float32)
    y_val = (X_val[:, 0] > 0).astype(int)

    model = XGBoostClassifierWrapper(params={"n_estimators": 10, "max_depth": 3, "n_jobs": 1})
    summary = model.train(X_train, y_train, X_val, y_val, optimize_threshold=True)

    assert model.is_trained
    assert summary["training_duration_seconds"] > 0
    assert 0.0 < model.threshold < 1.0

    probs = model.predict_proba(X_val)
    preds = model.predict(X_val)
    assert len(probs) == 40
    assert len(preds) == 40
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Test save and reload
    save_file = tmp_path / "test_xgb.json"
    model.save(str(save_file))
    reloaded = XGBoostClassifierWrapper()
    reloaded.load(str(save_file))
    assert reloaded.is_trained
    assert np.allclose(reloaded.predict_proba(X_val), probs)
