"""
Pre-game feature builder for NBA Analytics v3.

Works for both:
- completed games
- scheduled (future) games

It uses historical data from ingestion_snapshot.parquet
to compute rolling features for games being predicted.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class PreGameFeatureBuilder:
    """
    Build pre-game features for a set of games, using the
    canonical ingestion snapshot as history.
    """

    snapshot_path: Path = Path("data/ingestion/ingestion_snapshot.parquet")
    rolling_window: int = 5  # last N games per team

    # ---------------------------------------------------------
    # Load ingestion snapshot
    # ---------------------------------------------------------
    def _load_history(self) -> pd.DataFrame:
        if not self.snapshot_path.exists():
            raise FileNotFoundError(
                f"Ingestion snapshot not found: {self.snapshot_path}"
            )

        logger.info(f"Loading ingestion snapshot → {self.snapshot_path}")
        df = pd.read_parquet(self.snapshot_path)

        required_cols = {
            "game_id",
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "status",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Ingestion snapshot missing columns: {missing}")

        df["date"] = pd.to_datetime(df["date"])

        # Only use completed games for historical features
        df = df[df["status"] == "final"]

        return df

    # ---------------------------------------------------------
    # Convert games → team-level rows
    # ---------------------------------------------------------
    def _prepare_team_history(self, history: pd.DataFrame) -> pd.DataFrame:
        home = history[
            ["game_id", "date", "home_team", "home_score", "away_score"]
        ].copy()
        home["team"] = home["home_team"]
        home["points_for"] = home["home_score"]
        home["points_against"] = home["away_score"]
        home["is_home"] = 1
        home["win"] = (home["home_score"] > home["away_score"]).astype(int)

        away = history[
            ["game_id", "date", "away_team", "home_score", "away_score"]
        ].copy()
        away["team"] = away["away_team"]
        away["points_for"] = away["away_score"]
        away["points_against"] = away["home_score"]
        away["is_home"] = 0
        away["win"] = (away["away_score"] > away["home_score"]).astype(int)

        team_games = pd.concat([home, away], ignore_index=True)

        # Drop rows with missing scores (scheduled games)
        team_games = team_games.dropna(subset=["points_for", "points_against"])
        team_games = team_games.sort_values(["team", "date"])
        return team_games

    # ---------------------------------------------------------
    # Rolling stats
    # ---------------------------------------------------------
    def _compute_team_rolling_features(self, team_games: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling stats for each team over past games.
        Ensures rolling columns always exist, even if all values are NaN.
        """
        # Initialize columns
        team_games["roll_winrate"] = np.nan
        team_games["roll_pts_for"] = np.nan
        team_games["roll_pts_against"] = np.nan

        def roll(group: pd.DataFrame) -> pd.DataFrame:
            group = group.sort_values("date")
            group["roll_winrate"] = group["win"].rolling(self.rolling_window).mean()
            group["roll_pts_for"] = (
                group["points_for"].rolling(self.rolling_window).mean()
            )
            group["roll_pts_against"] = (
                group["points_against"].rolling(self.rolling_window).mean()
            )
            return group

        return team_games.groupby("team", group_keys=False).apply(roll)

    # ---------------------------------------------------------
    # Build features for a set of games
    # ---------------------------------------------------------
    def build_for_games(self, games: pd.DataFrame) -> pd.DataFrame:
        history = self._load_history()

        # Standardize date column
        games = games.copy()
        if "date" not in games.columns and "game_date" in games.columns:
            games = games.rename(columns={"game_date": "date"})
        if "date" not in games.columns:
            raise ValueError("Input games must have 'date' or 'game_date' column.")

        games["date"] = pd.to_datetime(games["date"])

        # Prepare per-team history with outcomes
        team_games = self._prepare_team_history(history)
        team_games = self._compute_team_rolling_features(team_games)

        # Keep only rolling columns
        team_games = team_games[
            ["team", "date", "roll_winrate", "roll_pts_for", "roll_pts_against"]
        ]

        feature_rows: List[dict] = []

        for _, row in games.iterrows():
            game_id = row["game_id"]
            game_date = row["date"]
            home_team = row["home_team"]
            away_team = row["away_team"]

            home_hist = team_games[
                (team_games["team"] == home_team) & (team_games["date"] < game_date)
            ]
            away_hist = team_games[
                (team_games["team"] == away_team) & (team_games["date"] < game_date)
            ]

            def last_row_or_nan(df: pd.DataFrame, col: str):
                return df[col].iloc[-1] if not df.empty else np.nan

            features = {
                "game_id": game_id,
                "date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_roll_winrate": last_row_or_nan(home_hist, "roll_winrate"),
                "home_roll_pts_for": last_row_or_nan(home_hist, "roll_pts_for"),
                "home_roll_pts_against": last_row_or_nan(home_hist, "roll_pts_against"),
                "away_roll_winrate": last_row_or_nan(away_hist, "roll_winrate"),
                "away_roll_pts_for": last_row_or_nan(away_hist, "roll_pts_for"),
                "away_roll_pts_against": last_row_or_nan(away_hist, "roll_pts_against"),
            }

            feature_rows.append(features)

        df_features = pd.DataFrame(feature_rows)

        # Attach raw scores for completed games
        score_lookup = history.set_index("game_id")[["home_score", "away_score"]]
        df_features = df_features.merge(
            score_lookup, left_on="game_id", right_index=True, how="left"
        )

        # Explicit column ordering
        ordered_cols = [
            "game_id",
            "date",
            "home_team",
            "away_team",
            "home_roll_winrate",
            "home_roll_pts_for",
            "home_roll_pts_against",
            "away_roll_winrate",
            "away_roll_pts_for",
            "away_roll_pts_against",
            "home_score",
            "away_score",
        ]
        df_features = df_features[ordered_cols]

        return df_features


# ------------------------------------------------------------
# Backwards-compatible wrapper
# ------------------------------------------------------------
def build_features(
    df_games: pd.DataFrame,
    snapshot_path: str = "data/ingestion/ingestion_snapshot.parquet",
) -> pd.DataFrame:
    builder = PreGameFeatureBuilder(snapshot_path=Path(snapshot_path))
    return builder.build_for_games(df_games)
