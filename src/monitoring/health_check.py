# ============================================================
# Project: NBA Analytics & Betting Engine
# Author: Sadiq
# Description: System-wide health checks for ingestion,
#              predictions, odds, bet logs, and model registry.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional

import pandas as pd
from loguru import logger

from src.config.paths import (
    SCHEDULE_SNAPSHOT,
    LONG_SNAPSHOT,
    ODDS_DIR,
    PREDICTIONS_DIR,
    LOGS_DIR,
)
from src.config.thresholds import MAX_DAYS_BACK


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------


@dataclass
class HealthIssue:
    level: str  # "error" | "warning" | "info"
    component: str  # e.g. "ingestion", "predictions", "odds"
    message: str
    details: Dict[str, Any]


@dataclass
class HealthReport:
    ok: bool
    issues: List[HealthIssue]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [asdict(i) for i in self.issues],
            "summary": self.summary,
        }


# ------------------------------------------------------------
# Health Checker
# ------------------------------------------------------------


class HealthChecker:
    """
    Performs system-wide health checks:
      - Ingestion freshness
      - Predictions freshness
      - Odds availability
      - Bet log integrity
      - Model registry presence
      - Filesystem checks
    """

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def run(
        self,
        target_date: Optional[date] = None,
    ) -> HealthReport:
        issues: List[HealthIssue] = []
        summary: Dict[str, Any] = {}

        if target_date is None:
            target_date = date.today()

        # 1. Ingestion health
        issues.extend(self._check_ingestion(summary))

        # 2. Predictions health
        issues.extend(self._check_predictions(target_date, summary))

        # 3. Odds health
        issues.extend(self._check_odds(target_date, summary))

        # 4. Bet log health
        issues.extend(self._check_bet_log(summary))

        # 5. Model registry health
        issues.extend(self._check_model_registry(summary))

        ok = not any(i.level == "error" for i in issues)

        report = HealthReport(ok=ok, issues=issues, summary=summary)
        self._log_report(report)

        return report

    # --------------------------------------------------------
    # Ingestion checks
    # --------------------------------------------------------
    def _check_ingestion(self, summary: Dict[str, Any]) -> List[HealthIssue]:
        issues = []

        # Schedule snapshot
        if not SCHEDULE_SNAPSHOT.exists():
            issues.append(
                HealthIssue(
                    level="error",
                    component="ingestion",
                    message="Schedule snapshot missing.",
                    details={"path": str(SCHEDULE_SNAPSHOT)},
                )
            )
            return issues

        schedule_df = pd.read_parquet(SCHEDULE_SNAPSHOT)
        summary["schedule_rows"] = len(schedule_df)

        if schedule_df.empty:
            issues.append(
                HealthIssue(
                    level="error",
                    component="ingestion",
                    message="Schedule snapshot is empty.",
                    details={},
                )
            )
            return issues

        # Freshness check
        max_date = pd.to_datetime(schedule_df["date"]).max().date()
        days_old = (date.today() - max_date).days
        summary["schedule_last_date"] = str(max_date)

        if days_old > MAX_DAYS_BACK:
            issues.append(
                HealthIssue(
                    level="warning",
                    component="ingestion",
                    message="Schedule snapshot appears stale.",
                    details={"days_old": days_old},
                )
            )

        # Long-format snapshot
        if not LONG_SNAPSHOT.exists():
            issues.append(
                HealthIssue(
                    level="error",
                    component="ingestion",
                    message="Long-format snapshot missing.",
                    details={"path": str(LONG_SNAPSHOT)},
                )
            )
            return issues

        long_df = pd.read_parquet(LONG_SNAPSHOT)
        summary["long_rows"] = len(long_df)

        if long_df.empty:
            issues.append(
                HealthIssue(
                    level="error",
                    component="ingestion",
                    message="Long-format snapshot is empty.",
                    details={},
                )
            )

        return issues

    # --------------------------------------------------------
    # Predictions checks
    # --------------------------------------------------------
    def _check_predictions(
        self,
        target_date: date,
        summary: Dict[str, Any],
    ) -> List[HealthIssue]:
        issues = []

        pattern = f"predictions_{target_date.isoformat()}_v*.parquet"
        files = sorted(PREDICTIONS_DIR.glob(pattern))

        if not files:
            issues.append(
                HealthIssue(
                    level="error",
                    component="predictions",
                    message="No predictions found for target date.",
                    details={"target_date": target_date.isoformat()},
                )
            )
            return issues

        latest = files[-1]
        preds = pd.read_parquet(latest)
        summary["predictions_file"] = latest.name
        summary["predictions_rows"] = len(preds)

        if preds.empty:
            issues.append(
                HealthIssue(
                    level="error",
                    component="predictions",
                    message="Predictions file is empty.",
                    details={"file": latest.name},
                )
            )

        # Basic sanity check
        if "win_probability" not in preds.columns:
            issues.append(
                HealthIssue(
                    level="error",
                    component="predictions",
                    message="Predictions missing win_probability column.",
                    details={"file": latest.name},
                )
            )

        return issues

    # --------------------------------------------------------
    # Odds checks
    # --------------------------------------------------------
    def _check_odds(
        self,
        target_date: date,
        summary: Dict[str, Any],
    ) -> List[HealthIssue]:
        issues = []

        odds_path = ODDS_DIR / f"odds_{target_date.isoformat()}.parquet"
        summary["odds_path"] = str(odds_path)

        if not odds_path.exists():
            issues.append(
                HealthIssue(
                    level="error",
                    component="odds",
                    message="Odds snapshot missing for target date.",
                    details={"path": str(odds_path)},
                )
            )
            return issues

        odds_df = pd.read_parquet(odds_path)
        summary["odds_rows"] = len(odds_df)

        if odds_df.empty:
            issues.append(
                HealthIssue(
                    level="error",
                    component="odds",
                    message="Odds snapshot is empty.",
                    details={"path": str(odds_path)},
                )
            )

        return issues

    # --------------------------------------------------------
    # Bet log checks
    # --------------------------------------------------------
    def _check_bet_log(self, summary: Dict[str, Any]) -> List[HealthIssue]:
        issues = []

        bet_path = LOGS_DIR / "bets.parquet"
        summary["bet_log_path"] = str(bet_path)

        if not bet_path.exists():
            issues.append(
                HealthIssue(
                    level="info",
                    component="bet_log",
                    message="Bet log not found (no bets placed yet).",
                    details={},
                )
            )
            return issues

        df = pd.read_parquet(bet_path)
        summary["bet_log_rows"] = len(df)

        if df.empty:
            issues.append(
                HealthIssue(
                    level="warning",
                    component="bet_log",
                    message="Bet log exists but is empty.",
                    details={},
                )
            )

        # Check for missing outcomes
        unresolved = df["won"].isna().sum()
        summary["unresolved_bets"] = int(unresolved)

        if unresolved > 50:
            issues.append(
                HealthIssue(
                    level="warning",
                    component="bet_log",
                    message="Large number of unresolved bets.",
                    details={"unresolved": unresolved},
                )
            )

        return issues

    # --------------------------------------------------------
    # Model registry checks
    # --------------------------------------------------------
    def _check_model_registry(self, summary: Dict[str, Any]) -> List[HealthIssue]:
        issues = []

        # For now, simply check that at least one model exists
        model_files = list(PREDICTIONS_DIR.glob("predictions_*_v*.parquet"))
        summary["model_versions_detected"] = len(model_files)

        if len(model_files) == 0:
            issues.append(
                HealthIssue(
                    level="warning",
                    component="model_registry",
                    message="No model versions detected in predictions directory.",
                    details={},
                )
            )

        return issues

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    def _log_report(self, report: HealthReport):
        if report.ok:
            logger.success("HealthChecker: system health OK.")
        else:
            logger.error("HealthChecker: system health issues detected.")

        for issue in report.issues:
            if issue.level == "error":
                logger.error(
                    f"[Health][{issue.component}] {issue.message} | {issue.details}"
                )
            elif issue.level == "warning":
                logger.warning(
                    f"[Health][{issue.component}] {issue.message} | {issue.details}"
                )
            else:
                logger.info(
                    f"[Health][{issue.component}] {issue.message} | {issue.details}"
                )


if __name__ == "__main__":
    checker = HealthChecker()
    report = checker.run()
    print(report.to_dict())
