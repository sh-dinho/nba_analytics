# ============================================================
# File: src/config_schema.py
# Purpose: Pydantic schema for NBA prediction pipeline configuration
# Project: nba_analysis
# Version: 1.0
#
# Dependencies:
# - pydantic
# - pydantic_settings
# - typing (standard library)
# ============================================================

from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Paths(BaseModel):
    raw: str
    cache: str
    history: str
    csv: str
    parquet: str
    logs: str
    models: str
    mlflow_artifacts: Optional[str] = None


class NBASettings(BaseModel):
    seasons: List[str]
    default_year: int
    players_min_points: int
    fetch_retries: int = 3
    retry_delay_ms: int = 1500


class ModelSettings(BaseModel):
    path: str = Field(default="models/nba_logreg.pkl", env="MODEL_PATH")
    type: str = Field(default="xgb", env="MODEL_TYPE")
    threshold: float = 0.5
    device: str = Field(default="cpu", env="DEVICE")
    feature_version: str = "v1"


class OutputSettings(BaseModel):
    save_csv: bool = True
    save_parquet: bool = False
    pretty_json: bool = True
    include_player_stats: bool = True
    include_betting_fields: bool = True


class LoggingSettings(BaseModel):
    level: str = Field(default="INFO", env="LOG_LEVEL")
    file: str = Field(default="data/logs/pipeline.log", env="LOG_FILE")


class MLflowSettings(BaseModel):
    enabled: bool = True
    experiment: str = "nba_predictions"
    run_prefix: str = "daily_prediction_"
    log_avg_probability: bool = True
    log_model_path: bool = True


class Config(BaseSettings):
    paths: Paths
    nba: NBASettings
    model: ModelSettings
    output: OutputSettings
    logging: LoggingSettings
    mlflow: MLflowSettings

    class Config:
        env_nested_delimiter = "__"  # e.g. MODEL__PATH overrides model.path


# -----------------------------
# Loader function
# -----------------------------
import yaml

def load_config(path: str = "config.yaml") -> Config:
    """Load and validate configuration from YAML + environment overrides."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)
