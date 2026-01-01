from __future__ import annotations
import pandas as pd
from loguru import logger
from datetime import timedelta


def run_ingestion_health_check(schedule_df: pd.DataFrame):
    """
    Comprehensive ingestion health check for TEAM-GAME canonical rows.
    Expected schema:
        game_id
        date
        team
        opponent
        is_home
        score
        opponent_score
        season
    """

    logger.info("🔍 Running ingestion health check (v4)...")

    # --------------------------------------------------------
    # 1. Schema check
    # --------------------------------------------------------
    required = {
        "game_id",
        "date",
        "team",
        "opponent",
        "is_home",
        "score",
        "opponent_score",
        "season",
    }

    missing = required - set(schedule_df.columns)
    if missing:
        logger.error(f"❌ Missing required columns: {missing}")
    else:
        logger.info("✔ Schema OK")

    # --------------------------------------------------------
    # 2. Schema version distribution
    # --------------------------------------------------------
    if "schema_version" in schedule_df.columns:
        dist = schedule_df["schema_version"].value_counts().to_dict()
        logger.info(f"📐 Schema versions present: {dist}")
    else:
        logger.warning("⚠ No schema_version column found")

    # --------------------------------------------------------
    # 3. Duplicate team-game rows
    # --------------------------------------------------------
    dupes = schedule_df[schedule_df.duplicated(subset=["game_id", "team"], keep=False)]
    if not dupes.empty:
        logger.warning(
            f"⚠ Duplicate (game_id, team) rows: "
            f"{dupes[['game_id','team']].drop_duplicates().to_dict(orient='records')}"
        )
    else:
        logger.info("✔ No duplicate team-game rows")

    # --------------------------------------------------------
    # 4. Missing game days (not missing calendar days)
    # --------------------------------------------------------
    schedule_df["date"] = pd.to_datetime(schedule_df["date"]).dt.date
    counts = schedule_df.groupby("date").size()

    min_d, max_d = counts.index.min(), counts.index.max()
    all_days = [min_d + timedelta(days=i) for i in range((max_d - min_d).days + 1)]

    missing_days = [
        d
        for d in all_days
        if d not in counts.index
        and (
            (d - timedelta(days=1)) in counts.index
            or (d + timedelta(days=1)) in counts.index
        )
    ]

    if missing_days:
        logger.warning(
            f"⚠ Missing {len(missing_days)} game days: {missing_days[:5]}..."
        )
    else:
        logger.info("✔ No missing game days")

    # --------------------------------------------------------
    # 5. Score sanity
    # --------------------------------------------------------
    if (schedule_df["score"] < 0).any() or (schedule_df["opponent_score"] < 0).any():
        logger.error("❌ Negative scores detected")
    else:
        logger.info("✔ Score sanity OK")

    # --------------------------------------------------------
    # 6. Pre-game rows (score NA)
    # --------------------------------------------------------
    pregame = schedule_df[
        schedule_df["score"].isna() | schedule_df["opponent_score"].isna()
    ]
    if not pregame.empty:
        logger.info(f"ℹ Found {len(pregame)} pre-game rows (scores missing)")
    else:
        logger.info("✔ No pre-game rows")

    # --------------------------------------------------------
    # 7. Home/away consistency
    # --------------------------------------------------------
    bad_home = schedule_df[
        (schedule_df["is_home"] == 1) & (schedule_df["team"] == schedule_df["opponent"])
    ]
    bad_away = schedule_df[
        (schedule_df["is_home"] == 0) & (schedule_df["team"] == schedule_df["opponent"])
    ]

    if not bad_home.empty or not bad_away.empty:
        logger.error("❌ Invalid home/away team assignments detected")
    else:
        logger.info("✔ Home/away assignments OK")

    # --------------------------------------------------------
    # 8. Game completeness (each game_id must have exactly 2 rows)
    # --------------------------------------------------------
    game_counts = schedule_df.groupby("game_id").size()
    bad_games = game_counts[game_counts != 2]

    if not bad_games.empty:
        logger.error(f"❌ Incomplete games detected: {bad_games.to_dict()}")
    else:
        logger.info("✔ Every game_id has exactly 2 team-game rows")

    # --------------------------------------------------------
    # 9. Opponent symmetry check
    # --------------------------------------------------------
    asym = schedule_df.groupby("game_id").apply(
        lambda g: set(g["team"]) != set(g["opponent"])
    )
    asym = asym[asym]

    if not asym.empty:
        logger.error(f"❌ Opponent mismatch in games: {asym.index.tolist()}")
    else:
        logger.info("✔ Opponent symmetry OK")

    # --------------------------------------------------------
    # 10. Score symmetry check
    # --------------------------------------------------------
    def score_mismatch(g):
        if len(g) != 2:
            return False
        a, b = g.iloc[0], g.iloc[1]
        return not (
            a["score"] == b["opponent_score"] and b["score"] == a["opponent_score"]
        )

    mismatches = schedule_df.groupby("game_id").apply(score_mismatch)
    mismatches = mismatches[mismatches]

    if not mismatches.empty:
        logger.error(f"❌ Score symmetry errors: {mismatches.index.tolist()}")
    else:
        logger.info("✔ Score symmetry OK")

    # --------------------------------------------------------
    # 11. Season inference sanity
    # --------------------------------------------------------
    season_counts = schedule_df["season"].value_counts().to_dict()
    logger.info(f"📅 Season distribution: {season_counts}")

    # --------------------------------------------------------
    # 12. API outage detection (days with 0 games)
    # --------------------------------------------------------
    zero_days = counts[counts == 0]
    if not zero_days.empty:
        logger.warning(
            f"⚠ Days with zero games (possible API outage): {zero_days.index.tolist()}"
        )

    logger.success("🏁 Ingestion health check complete.")
