"""Configuration loader for OcuTrace.

Loads settings from config/settings.yaml. All configurable values
are accessed through this module — no hardcoded values in other modules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

from src.paths import get_config_path

_DEFAULT_CONFIG_PATH = get_config_path()
_config_cache: Optional[dict[str, Any]] = None


def load_config(path: Optional[str] = None) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Parameters
    ----------
    path : str or None
        Path to YAML config file. If None, uses default
        config/settings.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_config() -> dict[str, Any]:
    """Get cached configuration (singleton).

    Returns
    -------
    dict
        Parsed configuration dictionary. Loaded once, cached thereafter.
    """
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache


def reset_config() -> None:
    """Reset cached config. Useful for testing."""
    global _config_cache
    _config_cache = None
