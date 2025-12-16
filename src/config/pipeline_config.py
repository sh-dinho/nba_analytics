# ============================================================
# File: src/config/pipeline_config.py
# Purpose: Configuration for NBA Analytics Pipeline
# ============================================================

import yaml
from pathlib import Path


class Config:
    def __init__(self, config_file="config.yaml"):
        # Load configuration from YAML file
        with open(config_file, "r") as file:
            config_data = yaml.safe_load(file)

        # Initialize configuration sections from the YAML, with defaults
        self.paths = {
            key: Path(value) for key, value in config_data.get("paths", {}).items()
        }
        self.nba = config_data.get(
            "nba",
            {
                "seasons": ["2022-23", "2023-24", "2024-25", "2025-26"],
                "default_year": 2025,
                "players_min_points": 20,
                "fetch_retries": 3,
                "retry_delay_ms": 1500,
            },
        )

        self.model = config_data.get(
            "model",
            {
                "path": "models/nba_logreg.pkl",
                "type": "xgb",
                "threshold": 0.5,
                "device": "cpu",
                "feature_version": "v1",
            },
        )

        self.output = config_data.get(
            "output",
            {
                "save_csv": True,
                "save_parquet": False,
                "pretty_json": True,
                "include_player_stats": True,
                "include_betting_fields": True,
            },
        )

        self.logging = config_data.get(
            "logging", {"level": "INFO", "file": "data/logs/pipeline.log"}
        )

        self.mlflow = config_data.get(
            "mlflow",
            {
                "enabled": True,
                "experiment": "nba_predictions",
                "run_prefix": "daily_prediction_",
                "log_avg_probability": True,
                "log_model_path": True,
                "tracking_uri": "http://localhost:5000",
                "artifact_location": "mlruns",
            },
        )

        self.betting = config_data.get("betting", {"threshold": 0.6})

        self.runner = config_data.get(
            "runner", {"shap_enabled": True, "commit_outputs": True}
        )

        self.database = config_data.get(
            "database",
            {
                "host": "localhost",
                "port": 5432,
                "user": "nba_user",
                "password": "secret_password",
            },
        )

        self.prediction = config_data.get(
            "prediction", {"lookahead_days": 3, "use_any_available": True}
        )

        self.schema_version = config_data.get("schema_version", "2.1")

        # Create missing directories for paths
        self.create_missing_dirs()

        # Optionally, validate the configuration
        self.validate_config()

    def __getattr__(self, item):
        """For safe attribute access."""
        try:
            return getattr(self, item)
        except AttributeError:
            raise AttributeError(f"Config object has no attribute '{item}'")

    def create_missing_dirs(self):
        """Ensure all necessary directories exist."""
        for path_key in self.paths.values():
            if not path_key.exists():
                path_key.mkdir(parents=True, exist_ok=True)

    def validate_config(self):
        """Validate the existence of essential configuration keys."""
        required_keys = {
            "paths",
            "nba",
            "model",
            "output",
            "logging",
            "mlflow",
            "betting",
            "runner",
            "database",
            "prediction",
            "schema_version",
        }
        missing_keys = required_keys - set(self.__dict__.keys())
        if missing_keys:
            raise ValueError(f"Missing required configuration sections: {missing_keys}")

        # Example validation: Make sure certain paths are valid
        for path_key in ["raw", "cache", "history"]:
            path = Path(
                self.paths.get(path_key, "")
            )  # Convert to Path here for validation
            if not path.exists():
                raise FileNotFoundError(
                    f"The path for '{path_key}' does not exist: {path}"
                )

    def get_abs_path(self, key):
        """Helper to get the absolute path of any directory."""
        return self.paths[key].resolve()  # Returns Path object
