from typing import List, Tuple
import numpy as np

def apply_feature_shift(
    X: np.ndarray,
    feature_names: List[str],
    shift_type: str = "packet_size_scale",
    severity: float = 1.5
) -> np.ndarray:
    """
    Applies controlled, documented distribution-shift perturbations to specific network flow features.
    Does NOT modify attack labels.
    """
    X_shifted = X.copy()
    
    if shift_type == "packet_size_scale":
        # Target packet size related columns
        target_indices = [
            i for i, name in enumerate(feature_names)
            if any(term in name.lower() for term in ["packet length", "pkt len", "bytes/s", "header length"])
        ]
        if target_indices:
            X_shifted[:, target_indices] = X_shifted[:, target_indices] * severity

    elif shift_type == "flow_duration_noise":
        # Target duration columns
        target_indices = [
            i for i, name in enumerate(feature_names)
            if "duration" in name.lower()
        ]
        if target_indices:
            noise = np.random.normal(0, severity, size=(len(X), len(target_indices)))
            X_shifted[:, target_indices] = np.maximum(0, X_shifted[:, target_indices] + noise)

    elif shift_type == "general_jitter":
        # Add subtle zero-mean Gaussian noise across all features
        noise = np.random.normal(0, severity * 0.1, size=X.shape)
        X_shifted = X_shifted + noise

    return X_shifted.astype(np.float32)
