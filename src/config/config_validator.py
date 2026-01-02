from __future__ import annotations

# ============================================================
# NBA Analytics Engine — Canonical Configuration Validator
# ============================================================

import os
import json
from pathlib import Path
from loguru import logger

from src.config.paths import (
    DATA_DIR,
    CANONICAL_DIR,
    FEATURES_DIR,
    MODEL_DIR,
    MODEL_REGISTRY_PATH,
    PREDICTIONS_DIR,
    ODDS_DIR,
    RESULTS_SNAPSHOT_DIR,
    BACKTEST_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    DASHBOARD_DIR,
    BET_LOG_DIR,
    STREAMLIT_APP_DIR,
    SNAPSHOTS_DIR,
    RAW_FEATURE_SNAPSHOT_PATH,
    SEASON_SCHEDULE_PATH,
)

from src.config.env import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    ODDS_API_KEY,
    FEATURE_FLAG_ENABLE_ALERTS,
    FEATURE_FLAG_ENABLE_MONITORING,
    FEATURE_FLAG_ENABLE_BACKTESTING,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _check_dir(path: Path, auto_create: bool) -> dict:
    exists = path.exists()
    if not exists and auto_create:
        logger.warning(f"[Config] Creating missing directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
        exists = True

    # Safer writability test
    writable = False
    if exists:
        try:
            test_file = path / ".write_test"
            test_file.write_text("ok")
            test_file.unlink()
            writable = True
        except Exception:
            writable = False

    return {
        "path": str(path),
        "exists": exists,
        "is_dir": path.is_dir(),
        "writable": writable,
    }


def _check_file(path: Path) -> dict:
    exists = path.exists()
    writable = False

    if exists:
        try:
            with open(path, "a"):
                writable = True
        except Exception:
            writable = False

    return {
        "path": str(path),
        "exists": exists,
        "is_file": path.is_file(),
        "writable": writable,
    }


def _check_env(name: str, value: str, required: bool) -> dict:
    return {
        "name": name,
        "set": bool(value),
        "required": required,
        "value_present": "***" if value else "",
    }


# ------------------------------------------------------------
# Main Validator
# ------------------------------------------------------------
def validate_config(auto_create_dirs: bool = False) -> dict:
    logger.info("[Config] Validating canonical configuration...")

    report = {
        "directories": {},
        "files": {},
        "environment": {},
        "feature_flags": {},
        "ok": True,
    }

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------
    dirs_to_check = {
        "DATA_DIR": DATA_DIR,
        "CANONICAL_DIR": CANONICAL_DIR,
        "FEATURES_DIR": FEATURES_DIR,
        "MODEL_DIR": MODEL_DIR,
        "PREDICTIONS_DIR": PREDICTIONS_DIR,
        "ODDS_DIR": ODDS_DIR,
        "RESULTS_SNAPSHOT_DIR": RESULTS_SNAPSHOT_DIR,
        "BACKTEST_DIR": BACKTEST_DIR,
        "REPORTS_DIR": REPORTS_DIR,
        "LOGS_DIR": LOGS_DIR,
        "DASHBOARD_DIR": DASHBOARD_DIR,
        "BET_LOG_DIR": BET_LOG_DIR,
        "STREAMLIT_APP_DIR": STREAMLIT_APP_DIR,
        "SNAPSHOTS_DIR": SNAPSHOTS_DIR,
    }

    for name, path in dirs_to_check.items():
        info = _check_dir(path, auto_create_dirs)
        report["directories"][name] = info
        if not info["exists"]:
            logger.error(f"[Config] Missing directory: {name} → {path}")
            report["ok"] = False

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------
    files_to_check = {
        "MODEL_REGISTRY_PATH": MODEL_REGISTRY_PATH,
        "RAW_FEATURE_SNAPSHOT_PATH": RAW_FEATURE_SNAPSHOT_PATH,
        "SEASON_SCHEDULE_PATH": SEASON_SCHEDULE_PATH,
    }

    for name, path in files_to_check.items():
        info = _check_file(path)
        report["files"][name] = info
        if not info["exists"]:
            logger.error(f"[Config] Missing file: {name} → {path}")
            report["ok"] = False

    # Validate model registry JSON
    if MODEL_REGISTRY_PATH.exists():
        try:
            data = json.loads(MODEL_REGISTRY_PATH.read_text())
            if "models" not in data:
                raise ValueError
        except Exception:
            logger.error("[Config] Model registry file is corrupted.")
            report["ok"] = False

    # --------------------------------------------------------
    # Environment Variables (dynamic requirements)
    # --------------------------------------------------------
    env_vars = {
        "TELEGRAM_BOT_TOKEN": (TELEGRAM_BOT_TOKEN, FEATURE_FLAG_ENABLE_ALERTS),
        "TELEGRAM_CHAT_ID": (TELEGRAM_CHAT_ID, FEATURE_FLAG_ENABLE_ALERTS),
        "ODDS_API_KEY": (ODDS_API_KEY, FEATURE_FLAG_ENABLE_MONITORING),
    }

    for name, (value, required) in env_vars.items():
        info = _check_env(name, value, required)
        report["environment"][name] = info

        if required and not info["set"]:
            logger.error(f"[Config] Missing required environment variable: {name}")
            report["ok"] = False
        elif not info["set"]:
            logger.warning(f"[Config] Optional env var not set: {name}")

    # --------------------------------------------------------
    # Feature Flags
    # --------------------------------------------------------
    report["feature_flags"] = {
        "ENABLE_ALERTS": FEATURE_FLAG_ENABLE_ALERTS,
        "ENABLE_MONITORING": FEATURE_FLAG_ENABLE_MONITORING,
        "ENABLE_BACKTESTING": FEATURE_FLAG_ENABLE_BACKTESTING,
    }

    logger.success("[Config] Configuration validation complete.")
    return report


# ------------------------------------------------------------
# Pretty Printer
# ------------------------------------------------------------
def print_config_report(report: dict):
    print("\n=== CANONICAL CONFIGURATION VALIDATION REPORT ===\n")

    print("Directories:")
    for name, info in report["directories"].items():
        status = "OK" if info["exists"] else "MISSING"
        print(f"  {name:<25} {status}  → {info['path']}")

    print("\nFiles:")
    for name, info in report["files"].items():
        status = "OK" if info["exists"] else "MISSING"
        print(f"  {name:<25} {status}  → {info['path']}")

    print("\nEnvironment Variables:")
    for name, info in report["environment"].items():
        status = "SET" if info["set"] else "MISSING"
        req = "(required)" if info["required"] else "(optional)"
        print(f"  {name:<25} {status:<8} {req}")

    print("\nFeature Flags:")
    for name, value in report["feature_flags"].items():
        print(f"  {name:<25} {value}")

    print("\nOverall Status:", "OK" if report["ok"] else "ISSUES DETECTED")
    print("\n=================================================\n")
