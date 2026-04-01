"""Configuration management for dc-property-impact."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_county_config(county_name: str = "loudoun") -> dict:
    """Load a county-specific YAML configuration."""
    config_path = CONFIGS_DIR / f"{county_name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
