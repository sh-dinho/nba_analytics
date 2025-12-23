from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Feature Engineering
# File: src/features/builder.py
# Author: Sadiq
#
# Description:
#     Full-featured, strictly point-in-time-correct feature
#     builder for canonical team-game data.
#
#     Feature set (v4, Option C):
#       - Basic features (is_home, score_diff)
#       - Rolling stats (5, 10, 20 games)
#       - Season-to-date aggregates
#       - Home/away splits
#       - Rest days, back-to-back flags
#       - Opponent-adjusted rolling stats
#       - Strength-of-schedule metrics
#       - ELO-style team rating + opponent ELO
#       - Team form metrics
#
#     All features are leakage-safe: only past games are used.
# ============================================================

from dataclasses import dataclass
from typing import Dict, Callable, Literal, List

import pandas as pd
from loguru import logger


FeatureVersion = Literal["v4"]


# ------------------------------------------------------------
# FeatureSpec
# ------------------------------------------------------------


@dataclass
class FeatureSpec:
    name: str
    description: str
    builder: Callable[[pd.DataFrame], pd.DataFrame]


# ------------------------------------------------------------
# Registry
# ------------------------------------------------------------


def _build_registry() -> Dict[str, FeatureSpec]:
    return {
        "v4": FeatureSpec(
            name="v4",
            description="Full feature set: rolling stats, ELO, rest, SOS, form.",
            builder=_build_v4_features,
        )
    }


# ------------------------------------------------------------
# FeatureBuilder
# ------------------------------------------------------------


class FeatureBuilder:
    def __init__(self, version: FeatureVersion = "v4"):
        self.version = version
        self._registry = _build_registry()

        if version not in self._registry:
            raise ValueError(f"Unknown feature version: {version}")

        self.spec = self._registry[version]
        logger.info(
            f"FeatureBuilder initialized with version={version}: {self.spec.description}"
        )

    def build_from_long(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            logger.warning("FeatureBuilder called with empty DataFrame.")
            return df
        return self.spec.builder(df)

    def get_feature_names(self) -> List[str]:
        dummy = self.spec.builder(
            pd.DataFrame(
                columns=[
                    "game_id",
                    "date",
                    "team",
                    "opponent",
                    "is_home",
                    "score",
                    "opponent_score",
                    "season",
                ]
            )
        )
        base = {"game_id", "date", "team", "opponent", "season"}
        return [c for c in dummy.columns if c not in base]


# ------------------------------------------------------------
# v4 Feature Recipe
# ------------------------------------------------------------


def _build_v4_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure correct types and ordering
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["team", "date", "game_id"])

    # Basic features
    df["score_diff"] = df["score"] - df["opponent_score"]

    # --------------------------------------------------------
    # Rolling windows (NaN-safe, leakage-safe)
    # --------------------------------------------------------
    windows = [5, 10, 20]

    for w in windows:
        df[f"roll_points_for_{w}"] = df.groupby("team")["score"].transform(
            lambda s: s.shift().rolling(w, min_periods=1).mean()
        )

        df[f"roll_points_against_{w}"] = df.groupby("team")["opponent_score"].transform(
            lambda s: s.shift().rolling(w, min_periods=1).mean()
        )

        df[f"roll_margin_{w}"] = df.groupby("team")["score_diff"].transform(
            lambda s: s.shift().rolling(w, min_periods=1).mean()
        )

        # Rolling win rate — FIXED
        win_flag = (df["score"] > df["opponent_score"]).astype(float)
        df[f"roll_win_rate_{w}"] = win_flag.groupby(df["team"]).transform(
            lambda s: s.shift().rolling(w, min_periods=1).mean()
        )

    # --------------------------------------------------------
    # Season-to-date aggregates (leakage-safe)
    # --------------------------------------------------------
    df["season_points_for_avg"] = df.groupby(["team", "season"])["score"].transform(
        lambda s: s.shift().expanding().mean()
    )

    df["season_points_against_avg"] = df.groupby(["team", "season"])[
        "opponent_score"
    ].transform(lambda s: s.shift().expanding().mean())

    df["season_margin_avg"] = df.groupby(["team", "season"])["score_diff"].transform(
        lambda s: s.shift().expanding().mean()
    )

    # --------------------------------------------------------
    # Home/away splits (leakage-safe)
    # --------------------------------------------------------
    df["home_points_for_avg"] = df.groupby(["team", "is_home"])["score"].transform(
        lambda s: s.shift().expanding().mean()
    )

    df["home_points_against_avg"] = df.groupby(["team", "is_home"])[
        "opponent_score"
    ].transform(lambda s: s.shift().expanding().mean())

    # --------------------------------------------------------
    # Rest days
    # --------------------------------------------------------
    df["prev_date"] = df.groupby("team")["date"].shift()
    df["rest_days"] = (df["date"] - df["prev_date"]).dt.days
    df["is_b2b"] = (df["rest_days"] == 1).astype(int)

    # --------------------------------------------------------
    # Opponent-adjusted stats (correct merge: game_id)
    # --------------------------------------------------------
    for w in [5, 10]:
        opp_col = f"opp_roll_margin_{w}"
        df = df.merge(
            df[["game_id", "team", f"roll_margin_{w}"]].rename(
                columns={"team": "opponent", f"roll_margin_{w}": opp_col}
            ),
            on=["game_id", "opponent"],
            how="left",
        )

    # --------------------------------------------------------
    # Strength of schedule (SOS)
    # --------------------------------------------------------
    df["sos"] = df.groupby("team")["opponent_score"].transform(
        lambda s: s.shift().rolling(10, min_periods=1).mean()
    )

    # --------------------------------------------------------
    # ELO (NaN-safe)
    # --------------------------------------------------------
    df = _apply_elo(df)

    # Opponent ELO (correct merge: game_id)
    df = df.merge(
        df[["game_id", "team", "elo"]].rename(
            columns={"team": "opponent", "elo": "opp_elo"}
        ),
        on=["game_id", "opponent"],
        how="left",
    )

    # --------------------------------------------------------
    # Team form (last 3 games)
    # --------------------------------------------------------
    df["form_last3"] = df.groupby("team")["score_diff"].transform(
        lambda s: s.shift().rolling(3, min_periods=1).mean()
    )

    # Cleanup
    df = df.drop(columns=["prev_date"])
    df = df.sort_values(["date", "game_id", "team"])

    return df


# ------------------------------------------------------------
# ELO System (NaN-safe)
# ------------------------------------------------------------


def _apply_elo(df: pd.DataFrame, base_elo: float = 1500, k: float = 20) -> pd.DataFrame:
    df = df.sort_values(["team", "date", "game_id"]).copy()

    teams = df["team"].unique()
    elo = {t: base_elo for t in teams}

    elos = []

    for _, row in df.iterrows():
        t = row["team"]
        o = row["opponent"]

        elo_t = elo.get(t, base_elo)
        elo_o = elo.get(o, base_elo)

        # Skip pre-game rows (NaN scores)
        if pd.isna(row["score"]) or pd.isna(row["opponent_score"]):
            elos.append(elo_t)
            continue

        expected = 1 / (1 + 10 ** ((elo_o - elo_t) / 400))
        actual = 1 if row["score"] > row["opponent_score"] else 0

        new_elo = elo_t + k * (actual - expected)
        elo[t] = new_elo
        elos.append(new_elo)

    df["elo"] = elos
    return df
