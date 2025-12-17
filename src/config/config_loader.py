# ============================================================
# File: src/config/config_loader.py
# Purpose: YAML config loader for pipeline v2.0
# Version: 2.0
# Author: Your Team
# Date: December 2025
# ============================================================

from pathlib import Path
import yaml


class Config:
    def __init__(self, path: str):
        with open(Path(path), "r") as f:
            self._config = yaml.safe_load(f)

    @property
    def logging(self):
        return self._config.get("logging", {})

    @property
    def betting(self):
        return self._config.get("betting", {})

    @property
    def strength(self):
        return self._config.get("strength", {})

    @property
    def model(self):
        return self._config.get("model", {})
