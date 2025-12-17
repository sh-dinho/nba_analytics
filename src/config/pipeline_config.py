# ============================================================
# File: src/config/pipeline_config.py
# Purpose: Configuration for NBA Analytics Pipeline
# ============================================================

from pathlib import Path
from typing import Dict, Any
import yaml
import logging
import os


SUPPORTED_SCHEMA_VERSIONS = {"2.1"}


class Paths:
    """
    Strongly-typed path container.
    Access paths as config.paths.cache, config.paths.history, etc.
    """

    def __init__(self, paths: Dict[str, str]):
        for key, value in paths.items():
            setattr(self, key, Path(value))

    def create_dirs(self):
        for value in self.__dict__.values():
            if isinstance(value, Path):
                value.mkdir(parents=True, exist_ok=True)


class Config:
    def __init__(self, config_file: str = "config.yaml") -> None:
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, "r") as f:
            try:
                config_data = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in {config_file}: {e}")

        # ---------------- PATHS ----------------
        self.paths = Paths(
            config_data.get(
                "paths",
                {
                    "raw": "data/raw",
                    "cache": "data/cache",
                    "history": "data/history",
                    "models": "models",
                    "logs": "data/logs",
                },
            )
        )
        self.paths.create_dirs()

        # ---------------- NBA ----------------
        self.nba: Dict[str, Any] = config_data.get(
            "nba",
            {
                "seasons": ["2022-23", "2023-24", "2024-25", "2025-26"],
                "default_year": 2025,
                "fetch_retries": 3,
                "retry_delay_ms": 1500,
            },
        )

        # ---------------- MODEL ----------------
        self.model: Dict[str, Any] = config_data.get(
            "model",
            {
                "type": "random_forest",
                "path": str(self.paths.models / "nba_random_forest.pkl"),
                "threshold": 0.6,
                "feature_version": "v1",
                "calibrated": True,
            },
        )

        # ---------------- OUTPUT ----------------
        self.output: Dict[str, Any] = config_data.get(
            "output",
            {
                "save_csv": True,
                "save_parquet": True,
                "pretty_json": True,
                "include_betting_fields": True,
            },
        )

        # ---------------- LOGGING ----------------
        self.logging: Dict[str, Any] = config_data.get(
            "logging",
            {
                "level": "INFO",
                "file": str(self.paths.logs / "pipeline.log"),
            },
        )
        self._configure_logging()

        # ---------------- MLFLOW ----------------
        self.mlflow: Dict[str, Any] = config_data.get(
            "mlflow",
            {
                "enabled": True,
                "experiment": "nba_predictions",
                "run_prefix": "daily_prediction_",
                "tracking_uri": "http://localhost:5000",
                "artifact_location": "mlruns",
            },
        )

        # ---------------- BETTING ----------------
        self.betting: Dict[str, Any] = config_data.get("betting", {"threshold": 0.6})

        # ---------------- RUNNER ----------------
        self.runner: Dict[str, Any] = config_data.get(
            "runner",
            {
                "shap_enabled": True,
                "commit_outputs": True,
            },
        )

        # ---------------- DATABASE ----------------
        self.database: Dict[str, Any] = config_data.get(
            "database",
            {
                "host": "localhost",
                "port": 5432,
                "user": "nba_user",
                "password": os.getenv("NBA_DB_PASSWORD"),
            },
        )

        # ---------------- PREDICTION ----------------
        self.prediction: Dict[str, Any] = config_data.get(
            "prediction",
            {
                "lookahead_days": 3,
                "use_any_available": True,
            },
        )

        # ---------------- SCHEMA ----------------
        self.schema_version: str = config_data.get("schema_version", "2.1")
        self._validate_schema()

    # ============================================================
    # Internal helpers
    # ============================================================

    def _validate_schema(self):
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported schema_version '{self.schema_version}'. "
                f"Supported versions: {SUPPORTED_SCHEMA_VERSIONS}"
            )

    def _configure_logging(self):
        level = getattr(logging, self.logging.get("level", "INFO").upper())
        log_file = self.logging.get("file")

        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )

    # ============================================================
    # Public helpers
    # ============================================================

    def get_model_path(self) -> Path:
        return Path(self.model["path"]).resolve()

    def get_master_schedule_path(self) -> Path:
        return self.paths.cache / "master_schedule.parquet"
