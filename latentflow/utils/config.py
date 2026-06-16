"""Config loading utilities.

Loads YAML configuration and converts to a nested namespace for easy access.
"""

from typing import Any, Dict, Optional
from pathlib import Path
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file.
    
    Args:
        config_path: Path to YAML config file.
        
    Returns:
        Dictionary containing config parameters.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    return config


def merge_configs(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override config into base config (recursive).
    
    Args:
        base_config: Base configuration dictionary.
        overrides: Override parameters (e.g., from CLI args).
        
    Returns:
        Merged configuration.
    """
    result = base_config.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """Simple namespace wrapper for config dictionary.
    
    Allows dot-notation access: config.model.depth instead of config['model']['depth'].
    """

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        """Initialize from dictionary, recursively converting nested dicts."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"Config({items})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
