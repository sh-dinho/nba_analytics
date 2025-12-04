# ============================================================
# File: scripts/merge_team_data.py
# Purpose: Merge team tables into one master table (teamdata_all)
# ============================================================

import argparse
import sqlite3
import pandas as pd
import datetime
from pathlib import Path

from core.paths import DATA_DIR, LOGS_DIR, ensure_dirs
from core.log_config import init_global_logger
from core.exceptions import PipelineError, FileError

logger = init_global_logger()

DB_PATH = DATA_DIR / "TeamData.sqlite"
MERGE_LOG = LOGS_DIR / "merge_team_data.log"


def get_current_season_label() -> str:
    """Determine current NBA season label based on today's date."""
    today = datetime.date.today()
    year = today.year
    if today.month >= 10:  # Oct–Dec → season spans current year → next year
        return f"{year}_{year+1}"
    else:  # Jan–Jun → season spans previous year → current year
        return f"{year-1}_{year}"


def merge_specific_season(con, season_label: str) -> pd.DataFrame | None:
    """Merge only a specific season table into teamdata_all."""
    season_table = f"teamdata_{season_label}"
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (season_table,))
    row = cursor.fetchone()

    if not row:
        logger.warning(f"No table found for season: {season_table}")
        return None

    logger.info(f"Found season table: {season_table}")
    df = pd.read_sql_query(f"SELECT * FROM {season_table}", con)
    df["Season"] = season_label
    return df


def merge_current_season(con) -> pd.DataFrame | None:
    """Merge only the current season table into teamdata_all."""
    season_label = get_current_season_label()
    return merge_specific_season(con, season_label)


def merge_all_seasons(con) -> pd.DataFrame | None:
    """Merge all season tables into teamdata_all."""
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'teamdata_%'")
    season_tables = [row[0] for row in cursor.fetchall()]

    if not season_tables:
        logger.warning("No season tables found in the database.")
        return None

    logger.info(f"Found {len(season_tables)} season tables: {season_tables}")
    frames = []
    for table in season_tables:
        df = pd.read_sql_query(f"SELECT * FROM {table}", con)
        df["Season"] = table.replace("teamdata_", "")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def merge_team_data(db_path: Path = DB_PATH,
                    use_all: bool = False,
                    season_filter: str | None = None,
                    export_json: bool = False) -> pd.DataFrame | None:
    """Merge season tables into teamdata_all, optionally exporting JSON."""
    ensure_dirs(strict=False)

    if not db_path.exists():
        raise FileError("TeamData database not found", file_path=str(db_path))

    try:
        con = sqlite3.connect(db_path)
    except Exception as e:
        raise PipelineError(f"Failed to connect to database {db_path}: {e}")

    try:
        if season_filter:
            master_df = merge_specific_season(con, season_filter)
            merge_mode = f"season={season_filter}"
        elif use_all:
            master_df = merge_all_seasons(con)
            merge_mode = "all"
        else:
            master_df = merge_current_season(con)
            merge_mode = "current"

        if master_df is None:
            return None

        master_df.to_sql("teamdata_all", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_teamdata_all_season ON teamdata_all(Season)")
        logger.info(f"✅ Merged into teamdata_all with {len(master_df)} rows")

        if export_json:
            out_json = DATA_DIR / "teamdata_all.json"
            master_df.to_json(out_json, orient="records", indent=2)
            logger.info(f"📑 Also exported merged data to {out_json}")
        else:
            out_json = None

        # Append summary log
        summary_entry = pd.DataFrame([{
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": merge_mode,
            "rows": len(master_df),
            "db_path": str(db_path),
            "json_export": str(out_json) if out_json else None,
        }])
        try:
            if MERGE_LOG.exists():
                summary_entry.to_csv(MERGE_LOG, mode="a", header=False, index=False)
            else:
                summary_entry.to_csv(MERGE_LOG, index=False)
            logger.info(f"📈 Merge summary appended to {MERGE_LOG}")
        except Exception as e:
            logger.warning(f"Failed to append merge summary: {e}")

        return master_df
    finally:
        con.close()


def print_latest_summary():
    """Print the latest merge summary entry without re-merging data."""
    if not MERGE_LOG.exists():
        logger.error("No merge summary log found.")
        return
    try:
        df = pd.read_csv(MERGE_LOG)
        if df.empty:
            logger.warning("Merge summary log is empty.")
            return
        latest = df.tail(1).iloc[0].to_dict()
        logger.info(f"📊 Latest merge summary: {latest}")
    except Exception as e:
        logger.error(f"Failed to read merge summary log: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge team tables into teamdata_all")
    parser.add_argument("--all", action="store_true", help="Merge all seasons instead of just current season")
    parser.add_argument("--season", type=str, default=None, help="Merge only a specific season (e.g. 2024_2025)")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Path to TeamData.sqlite database")
    parser.add_argument("--export-json", action="store_true", help="Also export merged master table to JSON format")
    parser.add_argument("--summary-only", action="store_true", help="Print the latest merge summary log entry without merging")
    args = parser.parse_args()

    if args.summary_only:
        print_latest_summary()
    else:
        merge_team_data(db_path=Path(args.db),
                        use_all=args.all,
                        season_filter=args.season,
                        export_json=args.export_json)
