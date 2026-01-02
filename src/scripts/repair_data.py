from __future__ import annotations

from datetime import date, timedelta
import pandas as pd
from loguru import logger

from src.ingestion.collector import fetch_scoreboard_for_date
from src.ingestion.normalizer.scoreboard_normalizer import normalize_scoreboard_to_wide
from src.ingestion.normalizer.wide_to_long import wide_to_long
from src.ingestion.normalizer.canonicalizer import canonicalize_team_game_df
from src.ingestion.validator.team_game_validator import validate_team_game_df
from src.ingestion.fallback.manager import FallbackManager
from src.ingestion.fallback.schedule_fallback import SeasonScheduleFallback

from src.config.paths import (
    DAILY_SCHEDULE_SNAPSHOT,
    LONG_SNAPSHOT,
)


def total_repair():
    logger.info("=== 🛠️ Starting Total Historical Repair (v5 Canonical) ===")

    start = date(2022, 10, 1)   # start of 2022–23 season
    end = date.today()

    all_wide = []
    all_long = []

    fallbacks = FallbackManager([SeasonScheduleFallback()])

    d = start
    while d <= end:
        logger.info(f"Fetching {d}...")

        raw = fetch_scoreboard_for_date(d)
        if raw.empty or "schema_version" in raw.columns:
            d += timedelta(days=1)
            continue

        # Normalize → wide → long
        try:
            wide = normalize_scoreboard_to_wide(raw)
            long = wide_to_long(wide)
        except Exception as e:
            logger.error(f"Normalization failed for {d}: {e}")
            d += timedelta(days=1)
            continue

        # Canonicalize
        try:
            long = canonicalize_team_game_df(long)
        except Exception as e:
            logger.error(f"Canonicalization failed for {d}: {e}")
            d += timedelta(days=1)
            continue

        # Apply fallbacks
        try:
            long = fallbacks.fill_missing_for_date(d, long)
        except Exception as e:
            logger.error(f"Fallback failed for {d}: {e}")
            d += timedelta(days=1)
            continue

        # Validate
        try:
            validate_team_game_df(long, raise_on_error=True)
        except Exception as e:
            logger.error(f"Validation failed for {d}: {e}")
            d += timedelta(days=1)
            continue

        all_wide.append(wide)
        all_long.append(long)

        d += timedelta(days=1)

    # Combine all days
    if not all_long:
        logger.error("No valid historical data collected.")
        return

    full_wide = pd.concat(all_wide, ignore_index=True)
    full_long = pd.concat(all_long, ignore_index=True)

    # Save snapshots
    DAILY_SCHEDULE_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    full_wide.to_parquet(DAILY_SCHEDULE_SNAPSHOT, index=False)
    full_long.to_parquet(LONG_SNAPSHOT, index=False)

    logger.success(
        f"REPAIR COMPLETE: {len(full_long)} canonical rows written "
        f"({len(full_wide)} wide rows)."
    )


if __name__ == "__main__":
    total_repair()