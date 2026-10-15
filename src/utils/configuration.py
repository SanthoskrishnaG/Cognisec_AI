from pathlib import Path
from typing import Any, Dict
import yaml

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file safely.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
