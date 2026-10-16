import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Set fixed random seeds across all libraries for deterministic execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

from src.utils.device import (
    get_device,
    get_device_str,
    get_device_info,
    get_xgboost_device,
    is_cuda_available,
    clear_gpu_memory,
)
