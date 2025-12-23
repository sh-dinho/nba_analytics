from __future__ import annotations
# ============================================================
# Project: NBA Analytics & Betting Engine
# Author: Sadiq
# Description: Monitor model performance, calibration, drift,
#              and real-world outcomes over time.
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


class ModelMonitor:
    """
    Monitors:
      - Calibration drift
      - Prediction distribution drift
      - Accuracy / Brier / LogLoss / AUC
      - Real-world outcomes (from bet logs)
      - Alerts when performance degrades
    """

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def run(
        self,
        predictions_df: Optional[pd.DataFrame] = None,
        outcomes_df: Optional[pd.DataFrame] = None,
    ) -> ModelMonitorReport:
        """
        predictions_df: must include columns:
            - win_probability
            - won (actual outcome)
            - model_version
            - date

        outcomes_df: optional, from bet logs (stake, result, ROI)
        """
        issues: List[ModelMonitorIssue] = []

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

        # Ensure required columns
        required = {"win_probability", "won", "model_version", "date"}
        missing = required - set(predictions_df.columns)
        if missing:
            issues.append(
                ModelMonitorIssue(
                    level="error",
                    message=f"Missing required prediction columns: {missing}",
                    details={"missing": list(missing)},
                )
            )
            return ModelMonitorReport(ok=False, issues=issues, metrics={})

        # Compute metrics
        metrics = self._compute_metrics(predictions_df, issues)

        # Optional: incorporate bet outcomes
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
        """
        Load the most recent predictions file from PREDICTIONS_DIR.
        """
        files = sorted(PREDICTIONS_DIR.glob("predictions_*_v*.parquet"))
        if not files:
            logger.warning("ModelMonitor: no prediction files found.")
            return None

        latest = files[-1]
        df = pd.read_parquet(latest)
        logger.info(f"ModelMonitor: loaded {len(df)} predictions from {latest.name}.")
        return df

    def _load_bet_outcomes(self) -> Optional[pd.DataFrame]:
        """
        Load bet log and compute outcomes if available.
        """
        path = LOGS_DIR / "bets.parquet"
        if not path.exists():
            logger.info("ModelMonitor: no bet log found.")
            return None

        df = pd.read_parquet(path)
        if df.empty:
            return None

        return df

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    def _compute_metrics(
        self,
        df: pd.DataFrame,
        issues: List[ModelMonitorIssue],
    ) -> Dict[str, Any]:
        """
        Compute calibration, accuracy, Brier, LogLoss, AUC, drift.
        """
        metrics: Dict[str, Any] = {}

        # Drop rows without outcomes
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

        # Brier score
        try:
            metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
        except Exception as e:
            issues.append(
                ModelMonitorIssue(
                    level="warning",
                    message="Failed to compute Brier score.",
                    details={"error": str(e)},
                )
            )

        # Log loss
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob, eps=1e-15))
        except Exception as e:
            issues.append(
                ModelMonitorIssue(
                    level="warning",
                    message="Failed to compute log loss.",
                    details={"error": str(e)},
                )
            )

        # AUC
        if len(np.unique(y_true)) == 2:
            try:
                metrics["auc"] = float(roc_auc_score(y_true, y_prob))
            except Exception as e:
                issues.append(
                    ModelMonitorIssue(
                        level="warning",
                        message="Failed to compute AUC.",
                        details={"error": str(e)},
                    )
                )
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
            issues.append(
                ModelMonitorIssue(
                    level="warning",
                    message="Failed to compute calibration curve.",
                    details={"error": str(e)},
                )
            )

        # Drift detection: compare prediction distribution to uniform baseline
        pred_mean = float(np.mean(y_prob))
        metrics["prediction_mean"] = pred_mean

        if pred_mean < 0.40 or pred_mean > 0.60:
            issues.append(
                ModelMonitorIssue(
                    level="warning",
                    message="Prediction distribution drift detected.",
                    details={"prediction_mean": pred_mean},
                )
            )

        return metrics

    # --------------------------------------------------------
    # Betting metrics
    # --------------------------------------------------------
    def _compute_betting_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute ROI, hit rate, and average stake from bet logs.
        """
        metrics: Dict[str, Any] = {}

        if "stake" not in df.columns or "ml" not in df.columns:
            return metrics

        # Compute returns
        def compute_return(row):
            if pd.isna(row.get("won")):
                return 0.0
            if row["won"] == 1:
                # Convert ML to decimal odds
                ml = row["ml"]
                if ml > 0:
                    dec = 1 + ml / 100.0
                else:
                    dec = 1 + 100.0 / abs(ml)
                return row["stake"] * (dec - 1)
            else:
                return -row["stake"]

        df["profit"] = df.apply(compute_return, axis=1)

        metrics["total_bets"] = int(len(df))
        metrics["total_staked"] = float(df["stake"].sum())
        metrics["total_profit"] = float(df["profit"].sum())

        if metrics["total_staked"] > 0:
            metrics["roi"] = metrics["total_profit"] / metrics["total_staked"]
        else:
            metrics["roi"] = None

        # Hit rate
        if "won" in df.columns:
            metrics["hit_rate"] = float(df["won"].mean())
        else:
            metrics["hit_rate"] = None

        return metrics

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    def _log_report(self, report: ModelMonitorReport):
        if report.ok:
            logger.success("ModelMonitor: all checks passed.")
        else:
            logger.error("ModelMonitor: issues detected in model performance.")

        for issue in report.issues:
            if issue.level == "error":
                logger.error(f"[ModelMonitor] {issue.message} | {issue.details}")
            elif issue.level == "warning":
                logger.warning(f"[ModelMonitor] {issue.message} | {issue.details}")
            else:
                logger.info(f"[ModelMonitor] {issue.message} | {issue.details}")

        logger.info(f"ModelMonitor metrics: {report.metrics}")


if __name__ == "__main__":
    monitor = ModelMonitor()
    monitor.run()
