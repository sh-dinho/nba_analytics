# ============================================================
# Project: NBA Analytics & Betting Engine
# Author: Sadiq
# Description: Admin API / service layer exposing orchestrator,
#              status, dashboard, and logs via FastAPI.
# ============================================================

from __future__ import annotations

from datetime import date, datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from loguru import logger
import pandas as pd

from src.pipeline.orchestrator import Orchestrator
from src.monitoring.system_status import SystemStatusAggregator
from src.monitoring.health_check import HealthChecker
from src.monitoring.model_monitor import ModelMonitor
from src.monitoring.dashboard_data import DashboardDataBuilder
from src.monitoring.bet_logger import BetLogger
from src.config.paths import PREDICTIONS_DIR
from src.alerts.alert_manager import AlertManager

# ------------------------------------------------------------
# App setup
# ------------------------------------------------------------

app = FastAPI(
    title="NBA Analytics & Betting Engine Admin API",
    version="1.0.0",
    description=(
        "Admin / internal API for orchestrating runs, inspecting system "
        "status, fetching dashboard data, and reviewing logs."
    ),
)

orchestrator = Orchestrator()
status_aggregator = SystemStatusAggregator()
health_checker = HealthChecker()
model_monitor = ModelMonitor()
dashboard_builder = DashboardDataBuilder()
bet_logger = BetLogger()
alert_manager = AlertManager()


# ------------------------------------------------------------
# Simple auth hook (placeholder)
# ------------------------------------------------------------


def get_auth_token(x_api_key: Optional[str] = Query(default=None, alias="x-api-key")):
    """
    Lightweight hook for future auth. Right now it just logs presence.
    Replace with real authentication when wiring to production.
    """
    if x_api_key is None:
        logger.warning(
            "Admin API called without x-api-key (auth disabled placeholder)."
        )
    return x_api_key


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def _parse_date_or_today(date_str: Optional[str]) -> date:
    if not date_str:
        return date.today()
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid date format: {date_str}. Use YYYY-MM-DD."
        )


# ============================================================
# Run endpoints
# ============================================================


@app.post("/run/daily")
def run_daily(
    target_date: Optional[str] = Query(
        default=None, description="Target date YYYY-MM-DD. Defaults to today."
    ),
    full_history_ingestion: bool = False,
    retrain_model: bool = False,
    run_betting: bool = True,
    dry_run_bets: bool = True,
    auth: str = Depends(get_auth_token),
):
    """
    Trigger a full daily orchestrator run.
    """
    d = _parse_date_or_today(target_date)
    logger.info(
        f"API /run/daily called for {d} | full_history={full_history_ingestion}, "
        f"retrain_model={retrain_model}, run_betting={run_betting}, dry_run={dry_run_bets}"
    )
    try:
        summary = orchestrator.run_daily(
            target_date=d,
            full_history_ingestion=full_history_ingestion,
            retrain_model=retrain_model,
            run_betting=run_betting,
            dry_run_bets=dry_run_bets,
        )
    except Exception as e:
        logger.exception(f"/run/daily failed: {e}")
        alert_manager.alert_error("admin_api.run_daily", str(e))
        raise HTTPException(status_code=500, detail=str(e))

    # Optionally send pipeline summary alert
    alert_manager.alert_pipeline_summary(summary=summary.__dict__)

    return JSONResponse(summary.__dict__)


# ============================================================
# Status endpoints
# ============================================================


@app.get("/status/system")
def get_system_status(
    target_date: Optional[str] = Query(
        default=None, description="Target date YYYY-MM-DD. Defaults to today."
    ),
    auth: str = Depends(get_auth_token),
):
    """
    Return unified system status snapshot.
    """
    d = _parse_date_or_today(target_date)
    status = status_aggregator.collect(target_date=d)
    return JSONResponse(status.to_dict())


@app.get("/status/health")
def get_health_status(
    target_date: Optional[str] = Query(
        default=None, description="Target date YYYY-MM-DD. Defaults to today."
    ),
    auth: str = Depends(get_auth_token),
):
    """
    Return system health check report only.
    """
    d = _parse_date_or_today(target_date)
    report = health_checker.run(target_date=d)
    return JSONResponse(report.to_dict())


@app.get("/status/model")
def get_model_status(
    auth: str = Depends(get_auth_token),
):
    """
    Return model monitoring report.
    """
    report = model_monitor.run()
    return JSONResponse(report.to_dict())


# ============================================================
# Dashboard endpoints
# ============================================================


@app.get("/dashboard")
def get_dashboard_data(
    auth: str = Depends(get_auth_token),
):
    """
    Return dashboard-ready aggregated metrics.
    """
    bundle = dashboard_builder.build()
    return JSONResponse(bundle.to_dict())


# ============================================================
# Bets / logs endpoints
# ============================================================


@app.get("/bets/log")
def get_bet_log(
    limit: int = Query(default=500, ge=1, le=10000),
    auth: str = Depends(get_auth_token),
):
    """
    Return recent bet log rows (most recent first).
    """
    df = bet_logger.load()
    if df.empty:
        return {"rows": []}

    df = df.sort_values("timestamp", ascending=False).head(limit)
    return {"rows": df.to_dict(orient="records")}


@app.get("/bets/summary/daily")
def get_daily_bet_summary(
    target_date: Optional[str] = Query(
        default=None, description="Prediction date YYYY-MM-DD. Defaults to today."
    ),
    auth: str = Depends(get_auth_token),
):
    """
    Return daily betting summary for a given prediction_date.
    """
    d = _parse_date_or_today(target_date)
    summary = bet_logger.daily_summary(d.isoformat())
    return summary


# ============================================================
# Predictions endpoints
# ============================================================


@app.get("/predictions/{target_date}")
def get_predictions_for_date(
    target_date: str,
    limit: int = Query(default=1000, ge=1, le=20000),
    auth: str = Depends(get_auth_token),
):
    """
    Return predictions for a given date (if available).
    """
    try:
        d = date.fromisoformat(target_date)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: {target_date}. Use YYYY-MM-DD.",
        )

    pattern = f"predictions_{d.isoformat()}_v*.parquet"
    files: List = sorted(PREDICTIONS_DIR.glob(pattern))
    if not files:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for {d.isoformat()}",
        )

    latest = files[-1]
    df = pd.read_parquet(latest)
    if df.empty:
        return {"file": latest.name, "rows": []}

    df = df.head(limit)
    return {
        "file": latest.name,
        "rows": df.to_dict(orient="records"),
    }


# ============================================================
# Alerts / utilities
# ============================================================


@app.post("/alerts/test")
def send_test_alert(
    message: Optional[str] = Query(
        default=None, description="Optional custom test message."
    ),
    auth: str = Depends(get_auth_token),
):
    """
    Send a test alert via AlertManager / Telegram to verify wiring.
    """
    text = message or f"Test alert from Admin API at {datetime.utcnow().isoformat()}."
    ok = alert_manager.alert("test", f"*Admin API Test Alert*\n\n{text}")
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to send test alert.")
    return {"status": "ok"}


# Root health
@app.get("/")
def root():
    return {
        "service": "NBA Analytics & Betting Engine Admin API",
        "version": "1.0.0",
        "time": datetime.utcnow().isoformat(),
    }
