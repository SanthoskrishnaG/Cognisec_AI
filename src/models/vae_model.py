import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.utils.device import get_device

class TabularVAE(nn.Module):
    """
    Variational Autoencoder architecture for tabular network flow features.
    Learns latent mean and log-variance representations of benign network traffic.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        encoder_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True
    ):
        super(TabularVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_hidden_dims = encoder_hidden_dims or [64, 32]
        self.decoder_hidden_dims = decoder_hidden_dims or [32, 64]
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Build Encoder layers
        encoder_layers = []
        curr_dim = input_dim
        for h_dim in self.encoder_hidden_dims:
            encoder_layers.append(nn.Linear(curr_dim, h_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
            if dropout_rate > 0.0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
        self.encoder_backbone = nn.Sequential(*encoder_layers)

        # Latent heads: Mean and Log-Variance
        self.fc_mu = nn.Linear(curr_dim, latent_dim)
        self.fc_logvar = nn.Linear(curr_dim, latent_dim)

        # Build Decoder layers
        decoder_layers = []
        curr_dim = latent_dim
        for h_dim in self.decoder_hidden_dims:
            decoder_layers.append(nn.Linear(curr_dim, h_dim))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.LeakyReLU(0.2))
            if dropout_rate > 0.0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
        decoder_layers.append(nn.Linear(curr_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Clamp logvar to prevent numerical explosion
        logvar = torch.clamp(logvar, min=-15.0, max=15.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        # In deterministic evaluation mode, use mean representation
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


class VAETrainer:
    """
    Encapsulates training, validation, early stopping, and anomaly scoring for TabularVAE.
    Strictly trained on benign network traffic only.
    """
    def __init__(
        self,
        model: TabularVAE,
        beta_kl: float = 0.005,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.beta_kl = beta_kl
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_history = {"train_loss": [], "val_loss": [], "train_recon": [], "train_kl": []}
        self.threshold: float = 0.0
        self.target_fpr: float = 1.0

    def to(self, device: Union[str, torch.device]) -> "VAETrainer":
        """Move the VAE model and trainer to the specified target device."""
        self.device = get_device(device)
        self.model.to(self.device)
        return self

    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")
        kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        total_loss = recon_loss + self.beta_kl * kl_div
        return total_loss, recon_loss, kl_div

    def fit(
        self,
        X_train_benign: np.ndarray,
        X_val_benign: Optional[np.ndarray] = None,
        epochs: int = 30,
        batch_size: int = 512,
        early_stopping_patience: int = 6
    ) -> Dict[str, Any]:
        """
        Trains VAE on benign samples only.
        """
        train_tensor = torch.tensor(X_train_benign, dtype=torch.float32)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        val_loader = None
        if X_val_benign is not None:
            val_tensor = torch.tensor(X_val_benign, dtype=torch.float32)
            val_dataset = TensorDataset(val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss_acc = 0.0
            recon_loss_acc = 0.0
            kl_loss_acc = 0.0
            n_batches = len(train_loader)

            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                self.optimizer.zero_grad()
                x_recon, mu, logvar = self.model(batch_x)
                loss, recon, kl = self.compute_loss(batch_x, x_recon, mu, logvar)

                # Catch NaN losses
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError("Numerical instability: NaN encountered during VAE training.")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                train_loss_acc += loss.item()
                recon_loss_acc += recon.item()
                kl_loss_acc += kl.item()

            avg_train_loss = train_loss_acc / n_batches
            self.loss_history["train_loss"].append(avg_train_loss)
            self.loss_history["train_recon"].append(recon_loss_acc / n_batches)
            self.loss_history["train_kl"].append(kl_loss_acc / n_batches)

            # Validation loss on benign validation set
            if val_loader:
                self.model.eval()
                val_loss_acc = 0.0
                with torch.no_grad():
                    for (v_batch_x,) in val_loader:
                        v_batch_x = v_batch_x.to(self.device)
                        v_recon, v_mu, v_logvar = self.model(v_batch_x)
                        v_loss, _, _ = self.compute_loss(v_batch_x, v_recon, v_mu, v_logvar)
                        val_loss_acc += v_loss.item()
                avg_val_loss = val_loss_acc / len(val_loader)
                self.loss_history["val_loss"].append(avg_val_loss)

                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        break

        # Load best weights if early stopping was used
        if best_state is not None:
            self.model.load_state_dict(best_state)

        training_duration = time.time() - start_time
        return {
            "epochs_completed": epoch,
            "final_train_loss": avg_train_loss,
            "best_val_loss": best_val_loss if val_loader else None,
            "training_duration_seconds": round(training_duration, 3)
        }

    def compute_anomaly_scores(self, X: Union[np.ndarray, torch.Tensor], batch_size: int = 1024) -> np.ndarray:
        """
        Computes sample-wise mean squared reconstruction error as anomaly score.
        Score = (1 / D) * sum((x_i - x_recon_i)^2)
        Executes on configured device (CUDA or CPU) with inference_mode and no gradient building.
        """
        self.model.eval()

        if isinstance(X, torch.Tensor):
            tensor_x = X.detach()
        else:
            tensor_x = torch.as_tensor(X, dtype=torch.float32)

        total_samples = tensor_x.shape[0]
        if total_samples == 0:
            return np.array([], dtype=np.float32)

        scores = []
        with torch.inference_mode():
            for i in range(0, total_samples, batch_size):
                batch_x = tensor_x[i : i + batch_size]
                if batch_x.device != self.device:
                    batch_x = batch_x.to(self.device)
                recon, _, _ = self.model(batch_x)
                # Feature-wise squared errors
                sq_errors = torch.mean((batch_x - recon) ** 2, dim=1)
                scores.append(sq_errors.cpu().numpy())

        return np.concatenate(scores, axis=0)

    def calibrate_threshold(self, X_val_benign: np.ndarray, target_fpr: float = 1.0) -> float:
        """
        Calibrates anomaly threshold strictly on benign validation scores to enforce
        a pre-declared false alarm tolerance (e.g. 1.0% FPR).
        Threshold = percentile(100 - target_fpr).
        """
        val_scores = self.compute_anomaly_scores(X_val_benign)
        percentile = 100.0 - target_fpr
        self.threshold = float(np.percentile(val_scores, percentile))
        self.target_fpr = target_fpr
        return self.threshold

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predicts binary classification: 0 = Benign, 1 = Attack/Anomaly.
        """
        thresh = threshold if threshold is not None else self.threshold
        scores = self.compute_anomaly_scores(X)
        return (scores >= thresh).astype(int)

    def save_checkpoint(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.model.input_dim,
            "latent_dim": self.model.latent_dim,
            "encoder_hidden_dims": self.model.encoder_hidden_dims,
            "decoder_hidden_dims": self.model.decoder_hidden_dims,
            "threshold": self.threshold,
            "target_fpr": self.target_fpr,
            "beta_kl": self.beta_kl
        }, str(path))

    @classmethod
    def load_checkpoint(cls, filepath: str, device: Optional[Union[str, torch.device]] = None) -> "VAETrainer":
        path = Path(filepath)
        target_device = get_device(device)
        checkpoint = torch.load(str(path), map_location=target_device)
        model = TabularVAE(
            input_dim=checkpoint["input_dim"],
            latent_dim=checkpoint["latent_dim"],
            encoder_hidden_dims=checkpoint["encoder_hidden_dims"],
            decoder_hidden_dims=checkpoint["decoder_hidden_dims"]
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        trainer = cls(model=model, beta_kl=checkpoint.get("beta_kl", 0.005), device=target_device)
        trainer.model.eval()
        trainer.threshold = checkpoint.get("threshold", 0.0)
        trainer.target_fpr = checkpoint.get("target_fpr", 1.0)
        return trainer
