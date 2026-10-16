"""
Cognisec AI - Shared Device Selection & Hardware Acceleration Utility
Provides unified device management, CUDA detection, memory tracking,
and graceful CPU fallbacks for PyTorch and XGBoost models.
"""

import gc
import logging
from typing import Any, Dict, Optional, Union
import torch

logger = logging.getLogger("cognisec.device")


def is_cuda_available() -> bool:
    """Check if CUDA is available in the current PyTorch environment."""
    return bool(torch.cuda.is_available())


def get_device(preferred_device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Selects torch.device with automatic CUDA detection and graceful CPU fallback.

    Parameters
    ----------
    preferred_device : Optional[Union[str, torch.device]]
        Desired device (e.g. "cuda", "cuda:0", "cpu", or None for auto).

    Returns
    -------
    torch.device
        Selected device object ('cuda' or 'cpu').
    """
    if preferred_device is not None:
        if isinstance(preferred_device, torch.device):
            dev_str = preferred_device.type
        else:
            dev_str = str(preferred_device).lower().strip()

        if dev_str.startswith("cuda"):
            if is_cuda_available():
                return torch.device(dev_str)
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        elif dev_str == "cpu":
            return torch.device("cpu")

    # Auto-detection
    if is_cuda_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_str(preferred_device: Optional[Union[str, torch.device]] = None) -> str:
    """Return device as a lowercase string ('cuda' or 'cpu')."""
    dev = get_device(preferred_device)
    return "cuda" if dev.type == "cuda" else "cpu"


def get_xgboost_device(preferred_device: Optional[str] = None) -> str:
    """
    Selects device for XGBoost classifier with CUDA support verification.
    Preserves graceful fallback to 'cpu' if CUDA is unavailable or unsupported.

    When preferred_device is 'cuda', verifies and enables CUDA acceleration.
    When preferred_device is None (auto):
    PyTorch 2.13.0+cu132 runs on CUDA 13.2 while standard XGBoost wheels bundle CUDA 12.
    To prevent driver-level CUDA context collisions in multi-threaded application servers,
    auto mode uses CPU for XGBoost (which takes ~1.5ms per batch with zero VRAM footprint)
    while dedicating the NVIDIA RTX 3050 GPU entirely to the PyTorch VAE.

    Parameters
    ----------
    preferred_device : Optional[str]
        Desired device ("cuda", "cpu", or None for auto).

    Returns
    -------
    str
        'cuda' if explicitly requested, supported, and available; otherwise 'cpu'.
    """
    if preferred_device is not None and str(preferred_device).lower() == "cuda":
        if is_cuda_available():
            try:
                import xgboost as xgb
                probe = xgb.XGBClassifier(n_estimators=1, max_depth=1, device="cuda")
                probe._Booster = None
                return "cuda"
            except Exception as e:
                logger.warning(f"XGBoost CUDA probe failed ({e}). Falling back to CPU.")
                return "cpu"
        return "cpu"

    return "cpu"


def get_device_info() -> Dict[str, Any]:
    """
    Returns comprehensive hardware acceleration and memory diagnostic metrics.
    """
    cuda_avail = is_cuda_available()
    info: Dict[str, Any] = {
        "cuda_available": cuda_avail,
        "selected_device": "cuda" if cuda_avail else "cpu",
        "device_count": torch.cuda.device_count() if cuda_avail else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "compute_capability": None,
        "vram_total_gb": 0.0,
        "vram_allocated_mb": 0.0,
        "vram_reserved_mb": 0.0,
        "vram_free_gb": 0.0
    }

    if cuda_avail:
        try:
            curr_dev = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(curr_dev)
            info["current_device_index"] = curr_dev
            info["compute_capability"] = f"{props.major}.{props.minor}"
            total_bytes = props.total_memory
            alloc_bytes = torch.cuda.memory_allocated(curr_dev)
            res_bytes = torch.cuda.memory_reserved(curr_dev)

            info["vram_total_gb"] = round(total_bytes / (1024 ** 3), 2)
            info["vram_allocated_mb"] = round(alloc_bytes / (1024 ** 2), 2)
            info["vram_reserved_mb"] = round(res_bytes / (1024 ** 2), 2)
            info["vram_free_gb"] = round((total_bytes - res_bytes) / (1024 ** 3), 2)
        except Exception as e:
            logger.warning(f"Error querying detailed CUDA properties: {e}")

    return info


def clear_gpu_memory() -> None:
    """Releases cached GPU memory back to the driver."""
    gc.collect()
    if is_cuda_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            logger.debug(f"GPU memory cleanup notice: {e}")
