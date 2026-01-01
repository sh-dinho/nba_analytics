from __future__ import annotations

import os
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
)

from src.config.env import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    ODDS_API_KEY,
    FEATURE_FLAG_ENABLE_ALERTS,
    FEATURE_FLAG_ENABLE_MONITORING,
    FEATURE_FLAG_ENABLE_BACKTESTING,
)

def _check_dir(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "writable": os.access(path, os.W_OK),
    }

def _check_file(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "writable": os.access(path, os.W_OK) if path.exists() else False,
    }

def _check_env(var_name: str, value: str) -> dict:
    return {
        "name": var_name,
        "set": bool(value),
        "value_present": "***" if value else "",
    }

def validate_config() -> dict:
    logger.info("Validating configuration...")

    report = {
        "directories": {},
        "files": {},
        "environment": {},
        "feature_flags": {},
        "ok": True,
    }

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
    }

    for name, path in dirs_to_check.items():
        info = _check_dir(path)
        report["directories"][name] = info
        if not info["exists"]:
            report["ok"] = False

    files_to_check = {
        "MODEL_REGISTRY_PATH": MODEL_REGISTRY_PATH,
    }

    for name, path in files_to_check.items():
        info = _check_file(path)
        report["files"][name] = info
        if not info["exists"]:
            report["ok"] = False

    env_vars = {
        "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
        "ODDS_API_KEY": ODDS_API_KEY,
    }

    for name, value in env_vars.items():
        info = _check_env(name, value)
        report["environment"][name] = info
        if not info["set"]:
            report["ok"] = False

    report["feature_flags"] = {
        "ENABLE_ALERTS": FEATURE_FLAG_ENABLE_ALERTS,
        "ENABLE_MONITORING": FEATURE_FLAG_ENABLE_MONITORING,
        "ENABLE_BACKTESTING": FEATURE_FLAG_ENABLE_BACKTESTING,
    }

    logger.success("Configuration validation complete.")
    return report

def print_config_report(report: dict):
    print("\n=== CONFIGURATION VALIDATION REPORT ===\n")

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
        print(f"  {name:<25} {status}")

    print("\nFeature Flags:")
    for name, value in report["feature_flags"].items():
        print(f"  {name:<25} {value}")

    print("\nOverall Status:", "OK" if report["ok"] else "ISSUES DETECTED")
    print("\n=======================================\n")