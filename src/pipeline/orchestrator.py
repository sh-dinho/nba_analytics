# ============================================================
# 🏀 NBA Analytics v3
# Module: Daily Orchestrator
# File: src/pipeline/orchestrator.py
# Author: Sadiq
#
# Description:
#     Coordinates the daily workflow:
#       1. Ingestion (yesterday + today)
#       2. Moneyline predictions
#       3. Totals predictions
#       4. Spread predictions
#       5. Betting pipeline sanity join (odds + predictions)
#       6. Unified recommendations (ML + O/U + ATS)
#       7. Notifications (run summary + recommendations)
#
#     Logs:
#       - model_name/model_version/feature_version used
#       - step-level success/failure + errors
# ============================================================

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from loguru import logger

from src.config.paths import (
    ODDS_DIR,
    PREDICTIONS_DIR,
    ORCHESTRATOR_LOG_DIR,
    DATA_DIR,
)
from src.ingestion.pipeline import run_today_ingestion
from src.model.predict import run_prediction_for_date
from src.model.predict_totals import run_totals_prediction_for_date
from src.model.predict_spread import run_spread_prediction_for_date
from src.alerts.telegram import send_telegram_message
from src.markets.recommend import generate_recommendations
from src.alerts.recommendations_alert import send_recommendations_alert

TOTALS_DIR = DATA_DIR / "predictions_totals"
SPREAD_DIR = DATA_DIR / "predictions_spread"


# ------------------------------------------------------------
# Data classes
# ------------------------------------------------------------


@dataclass
class StepResult:
    name: str
    status: str
    started_at: str
    ended_at: str
    error: Optional[str] = None
    extra: dict | None = None


@dataclass
class RunSummary:
    run_id: str
    run_date_utc: str
    target_date: str
    steps: list[StepResult]
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    feature_version: Optional[str] = None
    num_recommendations: Optional[int] = None


# ------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------


class Orchestrator:
    def __init__(self, notify: bool = True):
        ORCHESTRATOR_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.notify = notify

    def run_daily(self, target_date: Optional[date] = None):
        target_date = target_date or date.today()
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
        run_date_str = datetime.now(timezone.utc).isoformat()

        logger.info(f"🚀 Starting NBA Orchestrator for {target_date} (run_id={run_id})")

        steps: list[StepResult] = []
        pred_meta: dict | None = None
        recs_count: int | None = None

        # 1) Ingestion
        steps.append(
            self._run_step("ingestion_today", lambda: self._step_ingestion(target_date))
        )

        # 2) Moneyline prediction
        def pred_wrapper():
            nonlocal pred_meta
            pred_meta = self._step_predict_moneyline(target_date)
            return pred_meta

        steps.append(self._run_step("predict_moneyline", pred_wrapper))

        # 3) Totals prediction
        steps.append(
            self._run_step(
                "predict_totals", lambda: self._step_predict_totals(target_date)
            )
        )

        # 4) Spread prediction
        steps.append(
            self._run_step(
                "predict_spread", lambda: self._step_predict_spread(target_date)
            )
        )

        # 5) Betting pipeline sanity join
        steps.append(
            self._run_step(
                "betting_pipeline", lambda: self._step_betting_pipeline(target_date)
            )
        )

        # 6) Recommendations (ML + O/U + ATS)
        def recs_wrapper():
            nonlocal recs_count
            extra = self._step_recommendations(target_date)
            recs_count = extra.get("num_recommendations", 0) if extra else 0
            return extra

        steps.append(self._run_step("recommendations", recs_wrapper))

        summary = RunSummary(
            run_id=run_id,
            run_date_utc=run_date_str,
            target_date=str(target_date),
            steps=steps,
            model_name=pred_meta.get("model_name") if pred_meta else None,
            model_version=pred_meta.get("model_version") if pred_meta else None,
            feature_version=pred_meta.get("feature_version") if pred_meta else None,
            num_recommendations=recs_count,
        )

        self._log_run_summary(summary)
        self._maybe_notify(summary)

        logger.success(f"✅ Orchestration complete for {target_date} (run_id={run_id})")

    # --------------------------------------------------------
    # Generic step runner
    # --------------------------------------------------------
    def _run_step(self, name: str, func: Callable[[], Optional[object]]) -> StepResult:
        started_at = datetime.now(timezone.utc).isoformat()
        logger.info(f"▶ Step '{name}' started")

        error_msg: Optional[str] = None
        status = "success"
        extra: dict | None = None

        try:
            result = func()
            if isinstance(result, dict):
                extra = result
        except Exception as e:
            status = "failed"
            error_msg = str(e)
            logger.error(f"❌ Step '{name}' failed: {error_msg}")
        else:
            logger.info(f"✔ Step '{name}' completed successfully.")

        ended_at = datetime.now(timezone.utc).isoformat()

        return StepResult(
            name=name,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            error=error_msg,
            extra=extra,
        )

    # --------------------------------------------------------
    # Steps
    # --------------------------------------------------------
    def _step_ingestion(self, target_date: date) -> dict:
        """
        Ingest games for yesterday and today (relative to target_date).
        """
        run_today_ingestion(today=target_date)
        return {"target_date": str(target_date)}

    def _step_predict_moneyline(self, target_date: date) -> dict:
        """
        Run moneyline prediction pipeline for target_date and extract model metadata
        from the returned predictions dataframe.
        """
        pred_df = run_prediction_for_date(target_date)
        if pred_df is None or pred_df.empty:
            logger.warning(f"No moneyline predictions generated for {target_date}.")
            return {}

        meta_fields = {}
        for field in ["model_name", "model_version", "feature_version"]:
            if field in pred_df.columns:
                values = pred_df[field].dropna().unique().tolist()
                meta_fields[field] = values[0] if values else None

        logger.info(
            "Moneyline prediction used model/feature versions: "
            f"{meta_fields.get('model_name')} | "
            f"{meta_fields.get('model_version')} | "
            f"{meta_fields.get('feature_version')}"
        )

        return meta_fields

    def _step_predict_totals(self, target_date: date) -> dict:
        """
        Run totals prediction pipeline for target_date.
        """
        pred_df = run_totals_prediction_for_date(target_date)
        count = 0 if pred_df is None else len(pred_df)
        logger.info(f"[Totals] Generated {count} totals predictions for {target_date}")
        return {"totals_predictions": count}

    def _step_predict_spread(self, target_date: date) -> dict:
        """
        Run spread prediction pipeline for target_date.
        """
        pred_df = run_spread_prediction_for_date(target_date)
        count = 0 if pred_df is None else len(pred_df)
        logger.info(f"[Spread] Generated {count} spread predictions for {target_date}")
        return {"spread_predictions": count}

    def _step_betting_pipeline(self, target_date: date) -> dict:
        """
        Simple odds + moneyline predictions join to validate data flow and log availability.
        """
        odds_path = ODDS_DIR / f"odds_{target_date}.parquet"
        preds_path = PREDICTIONS_DIR / f"predictions_{target_date}.parquet"

        if not preds_path.exists():
            logger.warning(
                f"No moneyline predictions file for {target_date}. Skipping betting pipeline."
            )
            return {"bets": 0}

        preds_df = pd.read_parquet(preds_path)

        if not odds_path.exists():
            logger.warning(
                f"No odds file for {target_date}. Skipping betting pipeline."
            )
            return {"bets": 0, "has_predictions": len(preds_df)}

        odds_df = pd.read_parquet(odds_path)

        merged = odds_df.merge(
            preds_df,
            on=["game_id", "team"],
            how="inner",
            suffixes=("_odds", "_pred"),
        )

        if merged.empty:
            logger.warning(
                f"No matching odds+moneyline predictions rows for {target_date}."
            )
            return {
                "bets": 0,
                "has_predictions": len(preds_df),
                "has_odds": len(odds_df),
            }

        logger.info(
            f"Betting pipeline joined {len(merged)} rows from odds+moneyline predictions for {target_date}."
        )

        return {
            "bets": len(merged),
            "has_predictions": len(preds_df),
            "has_odds": len(odds_df),
        }

    def _step_recommendations(self, target_date: date) -> dict:
        """
        Generate unified ML + O/U + ATS recommendations for target_date
        and (optionally) send them via Telegram.
        """
        ml_path = PREDICTIONS_DIR / f"predictions_{target_date}.parquet"
        totals_path = TOTALS_DIR / f"totals_{target_date}.parquet"
        spread_path = SPREAD_DIR / f"spread_{target_date}.parquet"
        odds_path = ODDS_DIR / f"odds_{target_date}.parquet"

        if (
            not ml_path.exists()
            and not totals_path.exists()
            and not spread_path.exists()
        ):
            logger.warning("No predictions available for recommendations.")
            return {"num_recommendations": 0}

        ml = pd.read_parquet(ml_path) if ml_path.exists() else pd.DataFrame()
        totals = (
            pd.read_parquet(totals_path) if totals_path.exists() else pd.DataFrame()
        )
        spread = (
            pd.read_parquet(spread_path) if spread_path.exists() else pd.DataFrame()
        )
        odds = pd.read_parquet(odds_path) if odds_path.exists() else pd.DataFrame()

        recs = generate_recommendations(
            ml_df=ml,
            totals_df=totals,
            spread_df=spread,
            odds_df=odds,
            start_date=None,
            end_date=None,
        )

        num_recs = len(recs)
        logger.info(f"Generated {num_recs} betting recommendations for {target_date}.")

        if self.notify and num_recs > 0:
            try:
                send_recommendations_alert(recs, target_date)
            except Exception as e:
                logger.warning(f"Failed to send recommendations alert: {e}")

        return {"num_recommendations": num_recs}

    # --------------------------------------------------------
    # Logging / Notification
    # --------------------------------------------------------
    def _log_run_summary(self, summary: RunSummary):
        ORCHESTRATOR_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = ORCHESTRATOR_LOG_DIR / "orchestrator_runs.log"

        records = []
        if log_path.exists():
            existing = log_path.read_text().strip()
            if existing:
                for line in existing.splitlines():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        record = asdict(summary)
        records.append(record)

        with log_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        logger.info(f"Run summary appended to {log_path}")

    def _maybe_notify(self, summary: RunSummary):
        if not self.notify:
            return

        lines = [
            "🏀 *NBA Orchestrator Run*",
            f"Run ID: `{summary.run_id}`",
            f"Run date (UTC): `{summary.run_date_utc}`",
            f"Target date: `{summary.target_date}`",
        ]

        if summary.model_name:
            lines.append(
                f"Model: `{summary.model_name}` v`{summary.model_version}` "
                f"(features `{summary.feature_version}`)"
            )

        if summary.num_recommendations is not None:
            lines.append(f"Recommendations: `{summary.num_recommendations}`")

        lines.append("")
        lines.append("*Steps:*")
        for step in summary.steps:
            line = f"- {step.name}: *{step.status}*"
            if step.error:
                line += f" — `{step.error}`"
            lines.append(line)

        message = "\n".join(lines)

        try:
            send_telegram_message(message)
        except Exception as e:
            logger.warning(f"Summary notification failed: {e}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(description="NBA Analytics v3 Orchestrator")
    parser.add_argument(
        "--date", type=str, default=None, help="Target date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--no-notify", action="store_true", help="Disable Telegram notifications"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.date:
        target_date = date.fromisoformat(args.date)
    else:
        target_date = date.today()

    orch = Orchestrator(notify=not args.no_notify)
    orch.run_daily(target_date=target_date)
