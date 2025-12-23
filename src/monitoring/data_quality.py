from __future__ import annotations
# ============================================================
# Project: NBA Analytics & Betting Engine
# Author: Sadiq
# Description: Data quality checks for canonical schedule,
#              long-format data, and feature sets.
# ============================================================


from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config.paths import SCHEDULE_SNAPSHOT, LONG_SNAPSHOT
from src.config.features import REQUIRED_COLUMNS_LONG, BASE_FEATURES


@dataclass
class DataQualityIssue:
    level: str  # "error" | "warning" | "info"
    category: str  # e.g. "schedule", "long", "features"
    message: str
    details: Dict[str, Any]


@dataclass
class DataQualityReport:
    ok: bool
    issues: List[DataQualityIssue]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [asdict(i) for i in self.issues],
        }


class DataQualityChecker:
    """
    Runs data quality checks on:
      - canonical schedule snapshot
      - long-format snapshot
      - feature DataFrames
    """

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def run_all(
        self,
        schedule_df: Optional[pd.DataFrame] = None,
        long_df: Optional[pd.DataFrame] = None,
        features_df: Optional[pd.DataFrame] = None,
    ) -> DataQualityReport:
        issues: List[DataQualityIssue] = []

        # Load from disk if not passed
        if schedule_df is None:
            schedule_df = self._load_snapshot(SCHEDULE_SNAPSHOT, "schedule")
        if long_df is None:
            long_df = self._load_snapshot(LONG_SNAPSHOT, "long")

        # Schedule checks
        if schedule_df is not None and not schedule_df.empty:
            issues.extend(self._check_schedule(schedule_df))
        else:
            issues.append(
                DataQualityIssue(
                    level="error",
                    category="schedule",
                    message="Schedule snapshot is missing or empty.",
                    details={},
                )
            )

        # Long-format checks
        if long_df is not None and not long_df.empty:
            issues.extend(self._check_long(long_df))
        else:
            issues.append(
                DataQualityIssue(
                    level="error",
                    category="long",
                    message="Long-format snapshot is missing or empty.",
                    details={},
                )
            )

        # Feature checks (optional, only if provided)
        if features_df is not None and not features_df.empty:
            issues.extend(self._check_features(features_df))

        ok = not any(i.level == "error" for i in issues)

        report = DataQualityReport(ok=ok, issues=issues)
        self._log_report(report)

        return report

    # --------------------------------------------------------
    # Load helpers
    # --------------------------------------------------------
    def _load_snapshot(self, path, category: str) -> Optional[pd.DataFrame]:
        if not path.exists():
            logger.warning(
                f"DataQualityChecker: {category} snapshot not found at {path}."
            )
            return None

        df = pd.read_parquet(path)
        if df.empty:
            logger.warning(
                f"DataQualityChecker: {category} snapshot at {path} is empty."
            )
        else:
            logger.info(
                f"DataQualityChecker: loaded {len(df)} rows from {path} ({category})."
            )
        return df

    # --------------------------------------------------------
    # Schedule checks
    # --------------------------------------------------------
    def _check_schedule(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        issues: List[DataQualityIssue] = []

        required = {
            "game_id",
            "date",
            "season",
            "season_type",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "status",
        }
        missing = required - set(df.columns)
        if missing:
            issues.append(
                DataQualityIssue(
                    level="error",
                    category="schedule",
                    message=f"Missing required schedule columns: {missing}",
                    details={"missing_columns": list(missing)},
                )
            )
            return issues

        # Duplicate game_id
        dup = df["game_id"].duplicated().sum()
        if dup > 0:
            issues.append(
                DataQualityIssue(
                    level="error",
                    category="schedule",
                    message=f"Found {dup} duplicate game_id rows.",
                    details={"duplicates": int(dup)},
                )
            )

        # Invalid scores (negative values)
        invalid_scores = ((df["home_score"] < 0) | (df["away_score"] < 0)).sum()
        if invalid_scores > 0:
            issues.append(
                DataQualityIssue(
                    level="warning",
                    category="schedule",
                    message=f"Found {invalid_scores} rows with negative scores.",
                    details={"rows": int(invalid_scores)},
                )
            )

        # Final games with missing scores
        final_missing_scores = df[
            (df["status"] == "final")
            & (df["home_score"].isna() | df["away_score"].isna())
        ].shape[0]
        if final_missing_scores > 0:
            issues.append(
                DataQualityIssue(
                    level="error",
                    category="schedule",
                    message=f"{final_missing_scores} final games have missing scores.",
                    details={"rows": int(final_missing_scores)},
                )
            )

        return issues

    # --------------------------------------------------------
    # Long-format checks
    # --------------------------------------------------------
    def _check_long(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        issues: List[DataQualityIssue] = []

        missing = set(REQUIRED_COLUMNS_LONG) - set(df.columns)
        if missing:
            issues.append(
                DataQualityIssue(
                    level="error",
                    category="long",
                    message=f"Missing required long-format columns: {missing}",
                    details={"missing_columns": list(missing)},
                )
            )
            return issues

        # Nulls in critical columns
        critical = ["game_id", "date", "team", "opponent", "season", "game_number"]
        for col in critical:
            n_null = df[col].isna().sum()
            if n_null > 0:
                issues.append(
                    DataQualityIssue(
                        level="error",
                        category="long",
                        message=f"Column '{col}' has {n_null} nulls.",
                        details={"column": col, "null_count": int(n_null)},
                    )
                )

        # game_number monotonicity per team-season
        bad_game_order = 0
        for (team, season), g in df.groupby(["team", "season"]):
            gn = g.sort_values("date")["game_number"].values
            if not np.all(np.diff(gn) >= 0):
                bad_game_order += 1
        if bad_game_order > 0:
            issues.append(
                DataQualityIssue(
                    level="warning",
                    category="long",
                    message=f"Found {bad_game_order} team-season groups with non-monotonic game_number.",
                    details={"groups": int(bad_game_order)},
                )
            )

        return issues

    # --------------------------------------------------------
    # Feature checks
    # --------------------------------------------------------
    def _check_features(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        issues: List[DataQualityIssue] = []

        missing = set(BASE_FEATURES) - set(df.columns)
        if missing:
            issues.append(
                DataQualityIssue(
                    level="error",
                    category="features",
                    message=f"Missing required feature columns: {missing}",
                    details={"missing_columns": list(missing)},
                )
            )
            return issues

        # Nulls in feature columns
        for col in BASE_FEATURES:
            n_null = df[col].isna().sum()
            if n_null > 0:
                issues.append(
                    DataQualityIssue(
                        level="warning",
                        category="features",
                        message=f"Feature column '{col}' has {n_null} nulls.",
                        details={"column": col, "null_count": int(n_null)},
                    )
                )

        # Optional: basic distribution checks (extreme values)
        for col in BASE_FEATURES:
            series = df[col].dropna()
            if series.empty:
                continue
            mean, std = series.mean(), series.std()
            if std == 0:
                issues.append(
                    DataQualityIssue(
                        level="info",
                        category="features",
                        message=f"Feature column '{col}' has zero variance.",
                        details={"column": col, "mean": float(mean)},
                    )
                )

        return issues

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    def _log_report(self, report: DataQualityReport):
        if report.ok:
            logger.success("DataQualityChecker: all checks passed with no errors.")
        else:
            logger.error("DataQualityChecker: errors detected in data quality checks.")

        for issue in report.issues:
            if issue.level == "error":
                logger.error(
                    f"[DQ][{issue.category}] {issue.message} | details={issue.details}"
                )
            elif issue.level == "warning":
                logger.warning(
                    f"[DQ][{issue.category}] {issue.message} | details={issue.details}"
                )
            else:
                logger.info(
                    f"[DQ][{issue.category}] {issue.message} | details={issue.details}"
                )


if __name__ == "__main__":
    checker = DataQualityChecker()
    checker.run_all()
