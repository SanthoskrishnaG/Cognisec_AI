import numpy as np
import pytest
import torch
from src.models.vae_model import TabularVAE, VAETrainer

def test_tabular_vae_architecture():
    batch_size = 16
    input_dim = 20
    latent_dim = 8
    vae = TabularVAE(input_dim=input_dim, latent_dim=latent_dim, encoder_hidden_dims=[32, 16], decoder_hidden_dims=[16, 32])

    x = torch.randn(batch_size, input_dim)
    x_recon, mu, logvar = vae(x)

    assert x_recon.shape == (batch_size, input_dim)
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    assert not torch.isnan(x_recon).any()

def test_vae_trainer_fit_and_threshold(tmp_path):
    np.random.seed(42)
    torch.manual_seed(42)
    # Synthetic benign traffic
    X_train_benign = np.random.randn(200, 15).astype(np.float32)
    X_val_benign = np.random.randn(100, 15).astype(np.float32)

    vae = TabularVAE(input_dim=15, latent_dim=4, encoder_hidden_dims=[16], decoder_hidden_dims=[16])
    trainer = VAETrainer(model=vae, beta_kl=0.001, device="cpu")

    summary = trainer.fit(X_train_benign, X_val_benign, epochs=3, batch_size=32)
    assert summary["epochs_completed"] == 3
    assert summary["training_duration_seconds"] > 0

    # Calibrate threshold at 2% target FPR
    thresh = trainer.calibrate_threshold(X_val_benign, target_fpr=2.0)
    assert thresh > 0.0

    scores = trainer.compute_anomaly_scores(X_val_benign)
    assert len(scores) == 100
    preds = trainer.predict(X_val_benign)
    # Check that empirical false alarm rate is close to target (2%)
    empirical_fpr = np.mean(preds == 1)
    assert empirical_fpr <= 0.05

    # Checkpoint save and reload
    ckpt_path = tmp_path / "test_vae.pt"
    trainer.save_checkpoint(str(ckpt_path))
    reloaded = VAETrainer.load_checkpoint(str(ckpt_path), device="cpu")
    reloaded_scores = reloaded.compute_anomaly_scores(X_val_benign)
    assert np.allclose(scores, reloaded_scores, atol=1e-5)
