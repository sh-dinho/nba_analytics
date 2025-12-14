# ============================================================
# File: src/analytics/rankings.py
# Purpose: Track top teams/players, betting recommendations,
#          and winning streaks from prediction history + box scores
# Version: 4.4 (robust MLflow config, artifact logging, error handling)
# ============================================================

import os
import logging
from typing import Dict, Optional

import pandas as pd
import mlflow

from src.mlflow_setup import configure_mlflow
from src.config import load_config  # ✅ corrected import to match your unified config


class RankingsManager:
    def __init__(self, config_file: str = "config.yaml") -> None:
        # Load validated config
        self.cfg = load_config(config_file)

        # Unpack config sections
        self.paths = self.cfg.paths
        self.nba_cfg = self.cfg.nba
        self.output_cfg = self.cfg.output
        self.logging_cfg = self.cfg.logging
        self.mlflow_cfg = self.cfg.mlflow
        self.betting_cfg = getattr(self.cfg, "betting", None)

        # Ensure directories exist
        os.makedirs(self.paths.history, exist_ok=True)
        os.makedirs(self.paths.raw, exist_ok=True)
        os.makedirs(self.paths.analytics, exist_ok=True)

        # Use a named logger; avoid basicConfig clashes
        self.logger = logging.getLogger("analytics.rankings")
        if not self.logger.handlers:
            handler = logging.FileHandler(self.logging_cfg.file)
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            self.logger.addHandler(handler)
            self.logger.setLevel(getattr(logging, self.logging_cfg.level, logging.INFO))

        # Paths
        self.conf_map_file = os.path.join(self.paths.raw, "team_conference_map.csv")
        self.analytics_dir = self.paths.analytics

        # Configure MLflow (if enabled in config)
        self.mlflow_enabled = getattr(self.mlflow_cfg, "enabled", True)
        if self.mlflow_enabled:
            try:
                configure_mlflow(
                    tracking_uri=self.mlflow_cfg.tracking_uri,
                    experiment_name=self.mlflow_cfg.experiment,
                )
                self.logger.info(
                    "MLflow configured: tracking_uri=%s", self.mlflow_cfg.tracking_uri
                )
            except Exception as e:
                self.logger.error("Failed to configure MLflow: %s", e)
                self.mlflow_enabled = False

    # -----------------------------
    # BETTING RECOMMENDATIONS
    # -----------------------------
    def betting_recommendations(
        self, team_rankings: pd.DataFrame, threshold: Optional[float] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Produce bet/avoid recommendations based on win_pct and pred_accuracy columns.

        Args:
            team_rankings: DataFrame with at least 'win_pct' and 'pred_accuracy' columns.
            threshold: Optional override for the decision threshold in [0, 1].
                       If None, uses config.betting.threshold.

        Returns:
            dict: {'bet_on': DataFrame, 'avoid': DataFrame}
        """
        # Validate columns
        required_cols = {"win_pct", "pred_accuracy"}
        missing = required_cols - set(team_rankings.columns)
        if missing:
            msg = f"team_rankings missing required columns: {missing}"
            self.logger.error(msg)
            raise ValueError(msg)

        # Resolve threshold
        if threshold is None:
            if (
                not self.betting_cfg
                or getattr(self.betting_cfg, "threshold", None) is None
            ):
                msg = "Betting threshold not provided and not found in config."
                self.logger.error(msg)
                raise ValueError(msg)
            try:
                threshold = float(self.betting_cfg.threshold)
            except Exception:
                msg = f"Invalid threshold value in config: {self.betting_cfg.threshold}"
                self.logger.error(msg)
                raise ValueError(msg)

        # Validate threshold bounds
        if not (0.0 <= threshold <= 1.0):
            msg = f"Invalid threshold {threshold}. Must be within [0, 1]."
            self.logger.error(msg)
            raise ValueError(msg)

        # Create views
        bet_on = team_rankings[
            (team_rankings["win_pct"] > threshold)
            & (team_rankings["pred_accuracy"] > threshold)
        ].copy()

        avoid = team_rankings[
            (team_rankings["win_pct"] < threshold)
            | (team_rankings["pred_accuracy"] < threshold)
        ].copy()

        self.logger.info(
            "Betting recommendations: bet_on=%d, avoid=%d", len(bet_on), len(avoid)
        )

        if bet_on.empty:
            self.logger.warning("No teams meet bet_on criteria.")
        if avoid.empty:
            self.logger.warning("No teams meet avoid criteria.")

        # Log to MLflow (optional)
        self._log_to_mlflow_artifacts(
            bet_on, "bet_on", {"count": len(bet_on), "threshold": threshold}
        )
        self._log_to_mlflow_artifacts(
            avoid, "avoid", {"count": len(avoid), "threshold": threshold}
        )

        return {"bet_on": bet_on, "avoid": avoid}

    # -----------------------------
    # INTERNAL: MLflow logging helper
    # -----------------------------
    def _log_to_mlflow_artifacts(
        self, df: pd.DataFrame, name: str, metrics: Dict[str, float]
    ) -> None:
        """Log metrics and a CSV artifact to MLflow if enabled."""
        if not self.mlflow_enabled:
            return

        try:
            with mlflow.start_run(nested=True):
                # Metrics
                for k, v in metrics.items():
                    try:
                        mlflow.log_metric(f"{name}_{k}", float(v))
                    except Exception:
                        continue

                # Artifact: save dataframe to a temp CSV and log
                artifact_file = os.path.join(self.analytics_dir, f"{name}.csv")
                df.to_csv(artifact_file, index=False)
                mlflow.log_artifact(artifact_file, artifact_path=f"analytics/{name}")
        except Exception as e:
            self.logger.error("Failed to log to MLflow for %s: %s", name, e)
