from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Monitoring Configuration
# File: src/config/monitoring.py
# Author: Sadiq
#
# Description:
#     Central configuration for:
#       • drift detection thresholds
#       • sample requirements
#       • PSI bucket configuration
#       • alerting thresholds
#
#     Supports environment overrides via:
#       NBA_MONITORING_ALPHA
#       NBA_MONITORING_MIN_SAMPLES
#       NBA_MONITORING_PSI_BUCKETS
#       NBA_MONITORING_PSI_THRESHOLD
# ============================================================

from dataclasses import dataclass
import os


# ------------------------------------------------------------
# Dataclass for typed monitoring config
# ------------------------------------------------------------

@dataclass(frozen=True)
class MonitoringConfig:
    drift_alpha: float
    min_samples: int
    psi_buckets: int
    psi_threshold: float


# ------------------------------------------------------------
# Environment override helper
# ------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val is not None else default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val is not None else default


# ------------------------------------------------------------
# Default configuration (with env overrides)
# ------------------------------------------------------------

MONITORING = MonitoringConfig(
    drift_alpha=_env_float("NBA_MONITORING_ALPHA", 0.05),
    min_samples=_env_int("NBA_MONITORING_MIN_SAMPLES", 200),
    psi_buckets=_env_int("NBA_MONITORING_PSI_BUCKETS", 10),
    psi_threshold=_env_float("NBA_MONITORING_PSI_THRESHOLD", 0.2),
)