from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Data Quality Dashboard Generator
# File: src/scripts/generate_data_quality_dashboard.py
# Author: Sadiq
#
# Description:
#     Generates a unified data-quality dashboard by combining:
#       • Canonical long snapshot validation
#       • Feature validation (snapshot or dynamic)
#       • Drift detection (KS + PSI)
#       • Model performance monitoring
#
#     Output:
#       LOGS_DIR / dashboard_latest.json
# ============================================================

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from loguru import logger

from src.config.paths import (
    LONG_SNAPSHOT,
    FEATURES_SNAPSHOT,   # optional — may or may not exist
    LOGS_DIR,
)
from src.features.builder import FeatureBuilder
from src.ingestion.validator.checks import (
    find_asymmetry,
    find_score_mismatches,
    find_incomplete_games,
)
from src.monitoring.drift import ks_drift_report, psi_report
from src.monitoring.model_monitor import ModelMonitor


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _read_parquet_sample(path: Path, max_rows: int = 50) -> pd.DataFrame:
    """Read a small sample from a parquet file or directory."""
    if not path.exists():
        raise FileNotFoundError(f"Parquet path does not exist: {path}")

    if path.is_file():
        return pd.read_parquet(path).head(max_rows)

    files = list(path.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in directory: {path}")

    return pd.read_parquet(files[0]).head(max_rows)


# ------------------------------------------------------------
# Long Snapshot Validation
# ------------------------------------------------------------

def validate_long_snapshot() -> Dict[str, Any]:
    if not LONG_SNAPSHOT.exists():
        return {"ok": False, "error": "Missing canonical long snapshot"}

    try:
        df = pd.read_parquet(LONG_SNAPSHOT)
    except Exception as e:
        return {"ok": False, "error": f"Failed to read long snapshot: {e}"}

    required = ["game_id", "team", "opponent", "date"]
    missing = [c for c in required if c not in df.columns]

    duplicate_rows = int(df.duplicated(subset=["game_id", "team"]).sum())
    asym = find_asymmetry(df)
    mismatches = find_score_mismatches(df)
    incomplete = find_incomplete_games(df)

    return {
        "ok": len(missing) == 0 and len(df) > 0 and duplicate_rows == 0,
        "rows": len(df),
        "missing_columns": missing,
        "duplicate_team_game_rows": duplicate_rows,
        "asymmetry_games": list(asym),
        "score_mismatches": list(mismatches),
        "incomplete_games": list(incomplete.index),
    }


# ------------------------------------------------------------
# Feature Validation (Snapshot or Dynamic)
# ------------------------------------------------------------

def validate_features(feature_version: str, max_rows: int = 500) -> Dict[str, Any]:
    """
    Try to load a feature snapshot.
    If missing → build features dynamically from LONG_SNAPSHOT.
    """

    # 1. Try snapshot
    if FEATURES_SNAPSHOT.exists():
        try:
            df = _read_parquet_sample(FEATURES_SNAPSHOT, max_rows=max_rows)
            return {
                "ok": True,
                "source": "snapshot",
                "rows_sampled": len(df),
                "columns": list(df.columns),
            }
        except Exception as e:
            logger.warning(f"Feature snapshot failed: {e}")

    # 2. Fallback → dynamic feature building
    if not LONG_SNAPSHOT.exists():
        return {"ok": False, "error": "No long snapshot for dynamic feature building"}

    try:
        df_long = pd.read_parquet(LONG_SNAPSHOT)
        fb = FeatureBuilder(version=feature_version)
        features = fb.build(df_long)
        sample = features.head(max_rows)
        return {
            "ok": True,
            "source": "dynamic",
            "rows_sampled": len(sample),
            "columns": list(sample.columns),
        }
    except Exception as e:
        return {"ok": False, "error": f"Dynamic feature build failed: {e}"}


# ------------------------------------------------------------
# Drift Detection
# ------------------------------------------------------------

def compute_feature_drift(feature_version: str) -> Dict[str, Any]:
    """
    Drift detection using:
      - snapshot if available
      - dynamic features otherwise
    """

    # Load features (snapshot or dynamic)
    feat_report = validate_features(feature_version)
    if not feat_report.get("ok"):
        return {"ok": False, "error": "Cannot compute drift: feature load failed"}

    # Load full feature dataset
    if FEATURES_SNAPSHOT.exists():
        if FEATURES_SNAPSHOT.is_dir():
            parts = list(FEATURES_SNAPSHOT.glob("**/*.parquet"))
            df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        else:
            df = pd.read_parquet(FEATURES_SNAPSHOT)
    else:
        df_long = pd.read_parquet(LONG_SNAPSHOT)
        fb = FeatureBuilder(version=feature_version)
        df = fb.build(df_long)

    if "date" not in df.columns:
        return {"ok": False, "error": "Features missing 'date' column"}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        return {"ok": False, "error": "No valid feature dates"}

    max_date = df["date"].max()
    recent = df[df["date"] >= max_date - timedelta(days=7)]
    baseline = df[df["date"] < max_date - timedelta(days=30)]

    if len(recent) < 50 or len(baseline) < 200:
        return {
            "ok": False,
            "error": "Insufficient data for drift detection",
            "recent_rows": len(recent),
            "baseline_rows": len(baseline),
        }

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    ks = ks_drift_report(baseline, recent, numeric_cols)
    psi = psi_report(baseline, recent, numeric_cols)

    drifted = [
        c for c, v in ks.items()
        if isinstance(v, dict) and v.get("drift") == 1.0
    ]

    return {
        "ok": len(drifted) == 0,
        "numeric_columns": numeric_cols,
        "ks": ks,
        "psi": psi,
        "drifted_columns": drifted,
        "recent_rows": len(recent),
        "baseline_rows": len(baseline),
    }


# ------------------------------------------------------------
# Model Monitoring
# ------------------------------------------------------------

def compute_model_monitoring() -> Dict[str, Any]:
    try:
        monitor = ModelMonitor()
        report = monitor.run()
        return {"ok": True, "report": report.to_dict()}
    except Exception as e:
        return {"ok": False, "error": f"Model monitoring failed: {e}"}


# ------------------------------------------------------------
# Dashboard Generator
# ------------------------------------------------------------

def generate_dashboard(feature_version: str) -> None:
    logger.info("=== Generating Data Quality Dashboard ===")

    long_report = validate_long_snapshot()
    feature_report = validate_features(feature_version)
    drift_report_ = compute_feature_drift(feature_version)
    model_report = compute_model_monitoring()

    dashboard = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "long_snapshot": long_report,
        "feature_snapshot": feature_report,
        "feature_drift": drift_report_,
        "model_monitoring": model_report,
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOGS_DIR / "dashboard_latest.json"
    out_path.write_text(json.dumps(dashboard, indent=2))

    logger.success(f"Dashboard written to {out_path}")

    print("\n=== DASHBOARD SUMMARY ===")
    for section, result in dashboard.items():
        if isinstance(result, dict) and "ok" in result:
            status = "OK" if result["ok"] else "ISSUES"
            print(f"{section}: {status}")
    print("\n=== DONE ===")


if __name__ == "__main__":
    generate_dashboard(feature_version="v1")