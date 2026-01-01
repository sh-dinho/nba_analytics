from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Model Performance & Drift Monitor
# File: src/monitoring/model_monitor.py
# Author: Sadiq
#
# Description:
#     Unified monitoring engine for:
#       • Model performance (accuracy, Brier, log loss, AUC)
#       • Calibration diagnostics (ECE, reliability curve)
#       • Prediction drift (KS-test + PSI)
#       • Model version consistency
#       • Betting performance (ROI, hit rate)
#
#     Produces a JSON‑ready ModelMonitorReport for dashboards,
#     CI pipelines, and daily monitoring jobs.
# ============================================================

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    accuracy_score,
    roc_auc_score,
)

from src.config.paths import LOGS_DIR, PREDICTIONS_DIR
from src.monitoring.drift import ks_drift_report, psi_report
from src.config.monitoring import MONITORING

# ------------------------------------------------------------
# Data classes
# ------------------------------------------------------------

@dataclass
class ModelMonitorIssue:
    level: str  # "error" | "warning" | "info"
    message: str
    details: Dict[str, Any]


@dataclass
class ModelMonitorReport:
    ok: bool
    issues: List[ModelMonitorIssue]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [asdict(i) for i in self.issues],
            "metrics": self.metrics,
        }


# ------------------------------------------------------------
# Main Monitor
# ------------------------------------------------------------

class ModelMonitor:
    """
    Monitors:
      - Model performance (accuracy, Brier, log loss, AUC)
      - Calibration (ECE, reliability curve)
      - Prediction drift (KS-test + PSI)
      - Model version consistency
      - Betting performance (ROI, hit rate)
    """

    REQUIRED_COLUMNS = {"win_probability", "won", "model_version", "date"}

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def run(
        self,
        predictions_df: Optional[pd.DataFrame] = None,
        outcomes_df: Optional[pd.DataFrame] = None,
    ) -> ModelMonitorReport:

        issues: List[ModelMonitorIssue] = []

        # Load predictions if not provided
        if predictions_df is None:
            predictions_df = self._load_latest_predictions()

        if predictions_df is None or predictions_df.empty:
            issues.append(
                ModelMonitorIssue(
                    level="error",
                    message="No predictions available for monitoring.",
                    details={},
                )
            )
            return ModelMonitorReport(ok=False, issues=issues, metrics={})

        # Validate required columns
        missing = self.REQUIRED_COLUMNS - set(predictions_df.columns)
        if missing:
            issues.append(
                ModelMonitorIssue(
                    level="error",
                    message=f"Missing required prediction columns: {missing}",
                    details={"missing": list(missing)},
                )
            )
            return ModelMonitorReport(ok=False, issues=issues, metrics={})

        # Compute performance metrics
        metrics = self._compute_performance_metrics(predictions_df, issues)

        # Drift detection
        metrics.update(self._compute_drift(predictions_df, issues))

        # Model version consistency
        metrics.update(self._compute_version_consistency(predictions_df, issues))

        # Betting outcomes
        if outcomes_df is None:
            outcomes_df = self._load_bet_outcomes()

        if outcomes_df is not None and not outcomes_df.empty:
            metrics.update(self._compute_betting_metrics(outcomes_df))

        ok = not any(i.level == "error" for i in issues)

        report = ModelMonitorReport(ok=ok, issues=issues, metrics=metrics)
        self._log_report(report)

        return report

    # --------------------------------------------------------
    # Loaders
    # --------------------------------------------------------
    def _load_latest_predictions(self) -> Optional[pd.DataFrame]:
        files = sorted(PREDICTIONS_DIR.glob("predictions_*_v*.parquet"))
        if not files:
            logger.warning("ModelMonitor: no prediction files found.")
            return None

        latest = files[-1]
        df = pd.read_parquet(latest)
        logger.info(f"ModelMonitor: loaded {len(df)} predictions from {latest.name}.")
        return df

    def _load_bet_outcomes(self) -> Optional[pd.DataFrame]:
        path = LOGS_DIR / "bets.parquet"
        if not path.exists():
            logger.info("ModelMonitor: no bet log found.")
            return None

        df = pd.read_parquet(path)
        return df if not df.empty else None

    # --------------------------------------------------------
    # Performance Metrics
    # --------------------------------------------------------
    def _compute_performance_metrics(
        self,
        df: pd.DataFrame,
        issues: List[ModelMonitorIssue],
    ) -> Dict[str, Any]:

        metrics: Dict[str, Any] = {}

        df = df.dropna(subset=["won", "win_probability"]).copy()
        if df.empty:
            issues.append(
                ModelMonitorIssue(
                    level="error",
                    message="No rows with actual outcomes for monitoring.",
                    details={},
                )
            )
            return metrics

        y_true = df["won"].astype(int).values
        y_prob = df["win_probability"].astype(float).values

        # Accuracy
        metrics["accuracy"] = float(accuracy_score(y_true, (y_prob >= 0.5).astype(int)))

        # Brier
        try:
            metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
        except Exception as e:
            issues.append(ModelMonitorIssue("warning", "Failed to compute Brier score.", {"error": str(e)}))

        # Log loss
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob, eps=1e-15))
        except Exception as e:
            issues.append(ModelMonitorIssue("warning", "Failed to compute log loss.", {"error": str(e)}))

        # AUC
        if len(np.unique(y_true)) == 2:
            try:
                metrics["auc"] = float(roc_auc_score(y_true, y_prob))
            except Exception as e:
                issues.append(ModelMonitorIssue("warning", "Failed to compute AUC.", {"error": str(e)}))
        else:
            metrics["auc"] = None

        # Calibration curve
        try:
            frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
            metrics["calibration_curve"] = {
                "mean_pred": mean_pred.tolist(),
                "frac_pos": frac_pos.tolist(),
            }
        except Exception as e:
            issues.append(ModelMonitorIssue("warning", "Failed to compute calibration curve.", {"error": str(e)}))

        # Expected Calibration Error (ECE)
        try:
            ece = np.mean(np.abs(frac_pos - mean_pred))
            metrics["ece"] = float(ece)
        except Exception:
            metrics["ece"] = None

        return metrics

    # --------------------------------------------------------
    # Drift Detection
    # --------------------------------------------------------
    def _compute_drift(
        self,
        df: pd.DataFrame,
        issues: List[ModelMonitorIssue],
    ) -> Dict[str, Any]:

        metrics: Dict[str, Any] = {}

        # Compare recent predictions to a uniform baseline
        baseline = pd.DataFrame({"win_probability": np.random.uniform(0, 1, 5000)})
        recent = df[["win_probability"]].copy()

        ks = ks_drift_report(baseline, recent, ["win_probability"])
        psi = psi_report(baseline, recent, ["win_probability"])

        metrics["ks_drift"] = ks
        metrics["psi_drift"] = psi

        # Flag drift
        ks_flag = ks["win_probability"]["drift"]
        psi_value = psi["win_probability"]

        if ks_flag == 1.0:
            issues.append(
                ModelMonitorIssue(
                    level="warning",
                    message="KS-test drift detected in win_probability.",
                    details={"ks": ks["win_probability"]},
                )
            )

        if psi_value is not None and psi_value > MONITORING["psi_threshold"]:
            issues.append(
                ModelMonitorIssue(
                    level="warning",
                    message="PSI drift detected in win_probability.",
                    details={"psi": psi_value},
                )
            )

        return metrics

    # --------------------------------------------------------
    # Model Version Consistency
    # --------------------------------------------------------
    def _compute_version_consistency(
        self,
        df: pd.DataFrame,
        issues: List[ModelMonitorIssue],
    ) -> Dict[str, Any]:

        versions = df["model_version"].unique().tolist()
        metrics = {"model_versions": versions}

        if len(versions) > 1:
            issues.append(
                ModelMonitorIssue(
                    level="warning",
                    message="Multiple model versions detected in predictions.",
                    details={"versions": versions},
                )
            )

        return metrics

    # --------------------------------------------------------
    # Betting Metrics
    # --------------------------------------------------------
    def _compute_betting_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}

        if "roi" in df.columns:
            metrics["betting_roi"] = float(df["roi"].mean())

        if "won" in df.columns:
            metrics["betting_hit_rate"] = float(df["won"].mean())

        return metrics

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    def _log_report(self, report: ModelMonitorReport) -> None:
        if report.ok:
            logger.success("ModelMonitor: OK")
        else:
            logger.error("ModelMonitor: FAIL")

        for issue in report.issues:
            if issue.level == "error":
                logger.error(f"[{issue.level.upper()}] {issue.message} | {issue.details}")
            elif issue.level == "warning":
                logger.warning(f"[{issue.level.upper()}] {issue.message} | {issue.details}")
            else:
                logger.info(f"[{issue.level.upper()}] {issue.message} | {issue.details}")