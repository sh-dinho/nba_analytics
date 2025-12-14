# ============================================================
# File: src/config.py
# Purpose: Unified configuration loader for NBA analysis pipeline
# Project: nba_analysis
# Version: 2.1 (fixes error reporting, extra keys, MLflow defaults, DB validation)
# ============================================================

from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError, root_validator
from pydantic_settings import BaseSettings
import yaml


# -----------------------------
# PATH CONFIGURATION
# -----------------------------
class Paths(BaseModel):
    raw: str = "data/raw"
    cache: str = "data/cache"
    history: str = "data/history"
    csv: str = "data/csv"
    parquet: str = "data/parquet"
    logs: str = "data/logs"
    models: str = "models"
    mlflow_artifacts: Optional[str] = "mlruns"
    analytics: Optional[str] = "data/analytics"


# -----------------------------
# NBA SETTINGS
# -----------------------------
class NBASettings(BaseModel):
    seasons: List[str]
    default_year: int
    players_min_points: int = Field(..., gt=0)
    fetch_retries: int = Field(default=3, ge=1)
    retry_delay_ms: int = Field(default=1500, ge=0)


# -----------------------------
# MODEL SETTINGS
# -----------------------------
class ModelSettings(BaseModel):
    path: str = Field(default="models/nba_logreg.pkl", env="MODEL_PATH")
    type: str = Field(default="xgb", env="MODEL_TYPE")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    device: str = Field(default="cpu", env="DEVICE")
    feature_version: str = "v1"


# -----------------------------
# OUTPUT SETTINGS
# -----------------------------
class OutputSettings(BaseModel):
    save_csv: bool = True
    save_parquet: bool = False
    pretty_json: bool = True
    include_player_stats: bool = True
    include_betting_fields: bool = True


# -----------------------------
# LOGGING SETTINGS
# -----------------------------
class LoggingSettings(BaseModel):
    level: str = Field(default="INFO", env="LOG_LEVEL")
    file: str = Field(default="data/logs/pipeline.log", env="LOG_FILE")


# -----------------------------
# MLFLOW SETTINGS
# -----------------------------
class MLflowSettings(BaseModel):
    enabled: bool = True
    experiment: str = "nba_predictions"
    run_prefix: str = "daily_prediction_"
    log_avg_probability: bool = True
    log_model_path: bool = True
    tracking_uri: str = "file:./mlruns"  # explicit default
    artifact_location: Optional[str] = None


# -----------------------------
# BETTING SETTINGS
# -----------------------------
class BettingSettings(BaseModel):
    threshold: float = Field(default=0.55, ge=0.0, le=1.0)


# -----------------------------
# RUNNER SETTINGS
# -----------------------------
class RunnerSettings(BaseModel):
    shap_enabled: bool = False
    commit_outputs: bool = True


# -----------------------------
# DATABASE SETTINGS
# -----------------------------
class DatabaseSettings(BaseModel):
    host: Optional[str] = Field(default=None, env="DATABASE_HOST")
    port: Optional[int] = Field(default=None, env="DATABASE_PORT")
    user: Optional[str] = Field(default=None, env="DATABASE_USER")
    password: Optional[str] = Field(default=None, env="DATABASE_PASSWORD")

    @root_validator
    def check_consistency(cls, values):
        # If any field is set, require all fields
        if any(values.values()) and not all(values.values()):
            raise ValueError(
                "Database config must include host, port, user, and password together."
            )
        return values


# -----------------------------
# ROOT CONFIG
# -----------------------------
class Config(BaseSettings):
    paths: Paths
    nba: NBASettings
    model: ModelSettings
    output: OutputSettings
    logging: LoggingSettings
    mlflow: MLflowSettings
    betting: BettingSettings = BettingSettings()
    runner: RunnerSettings = RunnerSettings()
    database: DatabaseSettings = DatabaseSettings()
    NBA_API_KEY: Optional[str] = Field(default=None, env="NBA_API_KEY")

    class Config:
        env_nested_delimiter = "__"  # e.g. MODEL__PATH overrides model.path
        extra = "ignore"  # tolerate extra keys in YAML


# -----------------------------
# Loader function
# -----------------------------
def load_config(path: str = "config.yaml") -> Config:
    """
    Load and validate configuration from YAML + environment overrides.
    Raises RuntimeError with detailed validation errors if invalid.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    try:
        return Config(**data)
    except ValidationError as e:
        raise RuntimeError(f"Configuration error: {e.errors()}")
