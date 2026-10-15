from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch

class ModelInterpretability:
    """
    Provides feature attribution and error decomposition for both models:
    - XGBoost: Gini/gain-based feature importance ranking.
    - VAE: Feature-wise reconstruction error decomposition for attack traffic vs normal baseline.
    """
    def __init__(self, feature_names: List[str]):
        self.feature_names = list(feature_names)

    def explain_xgboost_features(self, xgb_wrapper: Any, top_k: int = 15) -> pd.DataFrame:
        """
        Extracts native feature importance from trained XGBoost model.
        """
        booster = xgb_wrapper.model.get_booster()
        score_dict = booster.get_score(importance_type="gain")

        # Map f0, f1... to original feature names
        data = []
        for i, name in enumerate(self.feature_names):
            f_key = f"f{i}"
            gain = score_dict.get(f_key, score_dict.get(name, 0.0))
            data.append({"Feature": name, "Importance_Gain": gain})

        df = pd.DataFrame(data).sort_values(by="Importance_Gain", ascending=False).reset_index(drop=True)
        return df.head(top_k)

    def explain_vae_reconstruction_errors(
        self,
        vae_trainer: Any,
        X_scaled: np.ndarray,
        y: np.ndarray,
        top_k: int = 15
    ) -> pd.DataFrame:
        """
        Computes feature-wise mean squared reconstruction error disaggregated
        between benign and attack flows to reveal what features trigger anomalies.
        """
        vae_trainer.model.eval()
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            recon, _, _ = vae_trainer.model(tensor_x)
            sq_err = (tensor_x - recon).pow(2).cpu().numpy()

        benign_mask = (y == 0)
        attack_mask = (y == 1)

        mean_err_benign = np.mean(sq_err[benign_mask], axis=0) if np.sum(benign_mask) > 0 else np.zeros(len(self.feature_names))
        mean_err_attack = np.mean(sq_err[attack_mask], axis=0) if np.sum(attack_mask) > 0 else np.zeros(len(self.feature_names))
        ratio = (mean_err_attack + 1e-6) / (mean_err_benign + 1e-6)

        df = pd.DataFrame({
            "Feature": self.feature_names,
            "Benign_Recon_MSE": mean_err_benign,
            "Attack_Recon_MSE": mean_err_attack,
            "Anomaly_Discrepancy_Ratio": ratio
        }).sort_values(by="Anomaly_Discrepancy_Ratio", ascending=False).reset_index(drop=True)

        return df.head(top_k)
