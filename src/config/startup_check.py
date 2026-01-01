from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Startup System Check
# File: src/config/startup_check.py
# Author: Sadiq
# ============================================================

import json
import requests
from loguru import logger

from src.config.config_validator import validate_config, print_config_report
from src.config.paths import (
    MODEL_REGISTRY_PATH,
    SCHEDULE_SNAPSHOT,
    LONG_SNAPSHOT,
    FEATURES_SNAPSHOT,
)
from src.config.env import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    NBA_API_HEADERS,
)

def check_model_registry() -> dict:
    try:
        data = json.loads(MODEL_REGISTRY_PATH.read_text())
        ok = isinstance(data, dict) and "models" in data
        return {
            "ok": ok,
            "count": len(data.get("models", [])) if ok else 0,
            "path": str(MODEL_REGISTRY_PATH),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "path": str(MODEL_REGISTRY_PATH)}

def check_snapshots() -> dict:
    snapshots = {
        "schedule_snapshot": SCHEDULE_SNAPSHOT.exists(),
        "long_snapshot": LONG_SNAPSHOT.exists(),
        "features_snapshot": FEATURES_SNAPSHOT.exists(),
    }
    return {"ok": all(snapshots.values()), "snapshots": snapshots}

def check_nba_api() -> dict:
    url = "https://stats.nba.com/stats/scoreboardv3?GameDate=2024-01-01"
    try:
        resp = requests.get(url, headers=NBA_API_HEADERS, timeout=5)
        return {"ok": resp.status_code == 200, "status": resp.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def check_telegram() -> dict:
    ok = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
    return {
        "ok": ok,
        "token_present": bool(TELEGRAM_BOT_TOKEN),
        "chat_id_present": bool(TELEGRAM_CHAT_ID),
    }

def run_startup_check() -> dict:
    logger.info("Running full startup system check...")

    report = {
        "config": validate_config(),
        "model_registry": check_model_registry(),
        "snapshots": check_snapshots(),
        "nba_api": check_nba_api(),
        "telegram": check_telegram(),
    }

    report["ok"] = (
        report["config"]["ok"]
        and report["model_registry"]["ok"]
        and report["snapshots"]["ok"]
        and report["nba_api"]["ok"]
        and report["telegram"]["ok"]
    )

    return report

def print_startup_report(report: dict):
    print("\n========== STARTUP SYSTEM CHECK ==========\n")

    print("Configuration:")
    print_config_report(report["config"])

    print("Model Registry:")
    print(f"  OK: {report['model_registry']['ok']}")
    print(f"  Models: {report['model_registry'].get('count', 0)}")
    print(f"  Path: {report['model_registry']['path']}\n")

    print("Snapshots:")
    for name, exists in report["snapshots"]["snapshots"].items():
        print(f"  {name:<25} {'OK' if exists else 'MISSING'}")
    print()

    print("NBA API Connectivity:")
    nba = report["nba_api"]
    if nba["ok"]:
        print(f"  OK (status {nba['status']})")
    else:
        print(f"  FAILED → {nba.get('error', nba.get('status'))}")
    print()

    print("Telegram:")
    tel = report["telegram"]
    print(f"  OK: {tel['ok']}")
    print(f"  Token Present: {tel['token_present']}")
    print(f"  Chat ID Present: {tel['chat_id_present']}\n")

    print("Overall System Status:", "OK" if report["ok"] else "ISSUES DETECTED")
    print("\n==========================================\n")