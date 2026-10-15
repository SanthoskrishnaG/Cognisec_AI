import json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from src.models.vae_model import TabularVAE, VAETrainer
from src.evaluation.metrics import calculate_classification_metrics
from src.utils.configuration import load_yaml_config

def train_vae_pipeline(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    config_path: str = "configs/vae.yaml"
) -> Dict[str, Any]:
    """
    Executes deep learning VAE training on benign traffic only.
    Calibrates anomaly threshold strictly on benign validation set.
    """
    cfg = load_yaml_config(config_path)
    arch_cfg = cfg.get("architecture", {})
    train_cfg = cfg.get("training", {})
    thresh_cfg = cfg.get("threshold_selection", {})
    save_path = cfg.get("model_save_path", "artifacts/models/vae_checkpoint.pt")

    # Isolate BENIGN traffic only for VAE training and thresholding
    train_benign_mask = (y_train == 0)
    val_benign_mask = (y_val == 0)

    X_train_benign = X_train_scaled[train_benign_mask]
    X_val_benign = X_val_scaled[val_benign_mask]

    input_dim = X_train_scaled.shape[1]
    latent_dim = arch_cfg.get("latent_dim", 16)
    enc_hidden = arch_cfg.get("encoder_hidden_dims", [64, 32])
    dec_hidden = arch_cfg.get("decoder_hidden_dims", [32, 64])
    dropout = arch_cfg.get("dropout_rate", 0.1)
    use_bn = arch_cfg.get("use_batch_norm", True)

    vae = TabularVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_hidden_dims=enc_hidden,
        decoder_hidden_dims=dec_hidden,
        dropout_rate=dropout,
        use_batch_norm=use_bn
    )

    trainer = VAETrainer(
        model=vae,
        beta_kl=train_cfg.get("beta_kl", 0.005),
        learning_rate=train_cfg.get("learning_rate", 0.001),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        device=train_cfg.get("device", "cpu")
    )

    train_summary = trainer.fit(
        X_train_benign=X_train_benign,
        X_val_benign=X_val_benign,
        epochs=train_cfg.get("epochs", 30),
        batch_size=train_cfg.get("batch_size", 512),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 6)
    )

    # Calibrate threshold strictly on benign validation records
    target_fpr = thresh_cfg.get("default_target_fpr", 1.0)
    threshold = trainer.calibrate_threshold(X_val_benign=X_val_benign, target_fpr=target_fpr)

    # Save checkpoint
    trainer.save_checkpoint(save_path)

    # Evaluate on full validation set (benign + validation attacks)
    val_scores = trainer.compute_anomaly_scores(X_val_scaled)
    val_preds = (val_scores >= threshold).astype(int)
    val_metrics = calculate_classification_metrics(y_true=y_val, y_pred=val_preds, y_score=val_scores)

    results = {
        "model_name": "Variational Autoencoder",
        "training_summary": train_summary,
        "calibrated_threshold": threshold,
        "target_benign_fpr": target_fpr,
        "validation_metrics": val_metrics,
        "saved_model_path": save_path
    }
    return results
