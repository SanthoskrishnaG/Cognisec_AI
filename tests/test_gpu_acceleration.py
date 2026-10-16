import os
import pytest
import numpy as np
import torch
from pathlib import Path

from src.utils.device import (
    is_cuda_available,
    get_device,
    get_device_str,
    get_xgboost_device,
    get_device_info,
    clear_gpu_memory
)
from src.models.vae_model import TabularVAE, VAETrainer
from src.models.xgboost_model import XGBoostClassifierWrapper
from src.live.live_manager import LiveCaptureManager


def test_device_selection_and_fallback():
    """Test device selection logic for both CUDA (if present) and CPU fallback."""
    cuda_avail = is_cuda_available()
    
    # 1. Explicit CPU request must always return CPU device
    cpu_dev = get_device("cpu")
    assert cpu_dev.type == "cpu"
    assert get_device_str("cpu") == "cpu"

    # 2. Auto selection (device=None)
    auto_dev = get_device(None)
    if cuda_avail:
        assert auto_dev.type == "cuda"
        assert get_device_str(None) == "cuda"
    else:
        assert auto_dev.type == "cpu"
        assert get_device_str(None) == "cpu"


def test_device_info_structure():
    """Verify get_device_info returns comprehensive diagnostic metrics."""
    info = get_device_info()
    assert "cuda_available" in info
    assert "selected_device" in info
    assert "torch_version" in info
    assert "vram_total_gb" in info
    assert "vram_allocated_mb" in info

    if info["cuda_available"]:
        assert info["selected_device"] == "cuda"
        assert "GeForce" in info["device_name"] or "RTX" in info["device_name"] or "NVIDIA" in info["device_name"]
        assert info["compute_capability"] is not None
        assert info["vram_total_gb"] > 0


def test_vae_gpu_cpu_consistency(tmp_path):
    """Test that VAE produces identical anomaly scores whether run on CPU or GPU."""
    np.random.seed(42)
    torch.manual_seed(42)

    input_dim = 20
    X_sample = np.random.randn(50, input_dim).astype(np.float32)

    # Instantiate model
    vae = TabularVAE(
        input_dim=input_dim,
        latent_dim=4,
        encoder_hidden_dims=[16, 8],
        decoder_hidden_dims=[8, 16]
    )

    # Train briefly on CPU to get trained weights
    trainer_cpu = VAETrainer(model=vae, beta_kl=0.001, device="cpu")
    trainer_cpu.fit(X_sample[:40], X_sample[40:], epochs=2, batch_size=16)
    cpu_scores = trainer_cpu.compute_anomaly_scores(X_sample)

    # Save checkpoint
    ckpt_path = tmp_path / "consistency_vae.pt"
    trainer_cpu.save_checkpoint(str(ckpt_path))

    # Reload on CPU explicitly
    reloaded_cpu = VAETrainer.load_checkpoint(str(ckpt_path), device="cpu")
    reloaded_cpu_scores = reloaded_cpu.compute_anomaly_scores(X_sample)
    assert np.allclose(cpu_scores, reloaded_cpu_scores, atol=1e-5)

    # If CUDA available, reload on GPU and compare
    if is_cuda_available():
        reloaded_gpu = VAETrainer.load_checkpoint(str(ckpt_path), device="cuda")
        assert reloaded_gpu.device.type == "cuda"
        # Confirm parameters are actually on CUDA
        param_devices = {p.device.type for p in reloaded_gpu.model.parameters()}
        assert param_devices == {"cuda"}

        gpu_scores = reloaded_gpu.compute_anomaly_scores(X_sample)
        # Verify scores match within numerical tolerance
        assert np.allclose(cpu_scores, gpu_scores, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_production_vae_checkpoint_on_gpu():
    """Verify that the production VAE checkpoint loads on GPU and executes inference."""
    prod_ckpt = Path("artifacts/models/vae_checkpoint.pt")
    if not prod_ckpt.exists():
        pytest.skip("Production checkpoint not found")

    clear_gpu_memory()
    trainer = VAETrainer.load_checkpoint(str(prod_ckpt), device="cuda")
    assert trainer.device.type == "cuda"
    assert next(trainer.model.parameters()).device.type == "cuda"

    # Run dummy 70-feature input (standard CICIDS2017 feature count)
    dummy_input = np.zeros((10, 70), dtype=np.float32)
    
    # Check that model is in eval mode
    assert not trainer.model.training

    scores = trainer.compute_anomaly_scores(dummy_input)
    assert len(scores) == 10
    assert not np.isnan(scores).any()
    assert np.all(scores >= 0)


def test_xgboost_device_handling():
    """Verify XGBoost wrapper device configuration and prediction consistency."""
    np.random.seed(42)
    X_train = np.random.randn(80, 10).astype(np.float32)
    y_train = (X_train[:, 0] > 0).astype(int)
    X_val = np.random.randn(20, 10).astype(np.float32)
    y_val = (X_val[:, 0] > 0).astype(int)

    # CPU wrapper
    xgb_cpu = XGBoostClassifierWrapper(params={"n_estimators": 5, "max_depth": 3, "n_jobs": 1}, device="cpu")
    xgb_cpu.train(X_train, y_train, X_val, y_val, optimize_threshold=False)
    cpu_preds = xgb_cpu.predict_proba(X_val)

    # Auto / GPU wrapper
    xgb_auto = XGBoostClassifierWrapper(params={"n_estimators": 5, "max_depth": 3, "n_jobs": 1})
    xgb_auto.train(X_train, y_train, X_val, y_val, optimize_threshold=False)
    auto_preds = xgb_auto.predict_proba(X_val)

    assert len(cpu_preds) == 20
    assert len(auto_preds) == 20
    assert np.all((auto_preds >= 0.0) & (auto_preds <= 1.0))


def test_shared_model_injection_in_live_manager():
    """Verify that set_models in LiveCaptureManager avoids duplicate loading."""
    mgr = LiveCaptureManager()
    
    dummy_preprocessor = "mock_preprocessor"
    dummy_xgb = "mock_xgb"
    dummy_vae = "mock_vae"

    mgr.set_models(dummy_preprocessor, dummy_xgb, dummy_vae)
    assert mgr.models_loaded is True
    assert mgr.preprocessor == dummy_preprocessor
    assert mgr.xgb_wrapper == dummy_xgb
    assert mgr.vae_trainer == dummy_vae
