"""Configuration loading utilities.

All tunable values live in configs/*.yaml — never hard-code thresholds,
paths, or model names inside stage modules. This keeps the pipeline
configurable without touching source code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


class ConfigError(Exception):
    """Raised when a config file is missing or malformed."""


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config file by name (without extension) from configs/.

    Args:
        name: config file stem, e.g. "perception" for configs/perception.yaml

    Returns:
        Parsed config as a dict.

    Raises:
        ConfigError: if the file is missing or is not valid YAML.
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse {path}: {exc}") from exc

    if data is None:
        raise ConfigError(f"Config file is empty: {path}")

    return data
