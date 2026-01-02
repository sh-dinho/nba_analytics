from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Monitoring Configuration
# File: src/config/monitoring.py
# Author: Sadiq
# ============================================================

from dataclasses import dataclass
import os
from loguru import logger


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
# Environment override helpers (safe parsing)
# ------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        logger.warning(f"[Monitoring] Invalid float for {name}: {raw!r}. Using default={default}.")
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        logger.warning(f"[Monitoring] Invalid int for {name}: {raw!r}. Using default={default}.")
        return default


# ------------------------------------------------------------
# Load + validate monitoring config
# ------------------------------------------------------------

drift_alpha = _env_float("NBA_MONITORING_ALPHA", 0.05)
if not (0 < drift_alpha < 1):
    logger.warning(f"[Monitoring] drift_alpha={drift_alpha} out of range (0,1). Resetting to 0.05.")
    drift_alpha = 0.05

min_samples = _env_int("NBA_MONITORING_MIN_SAMPLES", 200)
if min_samples < 1:
    logger.warning(f"[Monitoring] min_samples={min_samples} invalid. Resetting to 200.")
    min_samples = 200

psi_buckets = _env_int("NBA_MONITORING_PSI_BUCKETS", 10)
if psi_buckets < 2:
    logger.warning(f"[Monitoring] psi_buckets={psi_buckets} invalid (<2). Resetting to 10.")
    psi_buckets = 10

psi_threshold = _env_float("NBA_MONITORING_PSI_THRESHOLD", 0.2)
if not (0 < psi_threshold < 1):
    logger.warning(f"[Monitoring] psi_threshold={psi_threshold} out of range (0,1). Resetting to 0.2.")
    psi_threshold = 0.2


# ------------------------------------------------------------
# Final canonical monitoring config
# ------------------------------------------------------------

MONITORING = MonitoringConfig(
    drift_alpha=drift_alpha,
    min_samples=min_samples,
    psi_buckets=psi_buckets,
    psi_threshold=psi_threshold,
)
