# ============================================================
# File: scripts/fetch_season_data.py
# Purpose: Automate NBA season data ingestion using config.toml
# ============================================================

import argparse
import toml
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

from core.paths import DATA_DIR, LOGS_DIR, ensure_dirs
from core.log_config import init_global_logger
from core.exceptions import PipelineError, DataError, FileError
from core.utils import ensure_columns

CONFIG_PATH = "config.toml"
OUTPUT_DIR = DATA_DIR / "seasons"
SEASON_LOG = LOGS_DIR / "season_data.log"

logger = init_global_logger()


def build_url(template: str, start_date: str, end_date: str,
              start_year: str, end_year: str, season_label: str) -> str:
    """Fill placeholders in data_url template."""
    try:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        end_year_str, end_month, end_day = end_dt.strftime("%Y-%m-%d").split("-")
    except ValueError:
        raise DataError(f"Invalid end_date format: {end_date}. Expected YYYY-MM-DD.")

    return template.format(end_month, end_day, start_year, end_year, season_label)


def fetch_season_data(config_path: str = CONFIG_PATH,
                      season_filter: str | None = None,
                      export_json: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch NBA season data based on config.toml settings. Returns dict of DataFrames."""
    ensure_dirs(strict=False)

    try:
        config = toml.load(config_path)
    except Exception as e:
        raise PipelineError(f"Failed to load config file {config_path}: {e}")

    url_template = config.get("data_url")
    if not url_template:
        raise DataError("Missing 'data_url' in config.toml")

    seasons = config.get("get-data", {})
    if not seasons:
        logger.warning("No seasons defined in config.toml")
        return {}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}

    for season, params in seasons.items():
        if season_filter and season != season_filter:
            continue

        season_label = params.get("season_label")
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        start_year = params.get("start_year")
        end_year = params.get("end_year")

        if not all([season_label, start_date, end_date, start_year, end_year]):
            logger.warning(f"⚠️ Skipping {season}: missing parameters in config.")
            continue

        url = build_url(url_template, start_date, end_date, start_year, end_year, season_label)
        logger.info(f"Fetching {season_label} → {url}")

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nba.com/",
            "Accept": "application/json"
        }

        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"❌ Failed to fetch {season_label}: {e}")
            continue

        try:
            data = r.json()
            rows = data["resultSets"][0]["rowSet"]
            headers = data["resultSets"][0]["headers"]
            df = pd.DataFrame(rows, columns=headers)

            ensure_columns(df, {"TEAM_ID", "TEAM_NAME"}, f"{season_label} season data")

            safe_label = season_label.replace(" ", "_")
            out_csv = OUTPUT_DIR / f"teamdata_{safe_label}.csv"
            df.to_csv(out_csv, index=False)
            logger.info(f"✅ Saved {season_label} data to {out_csv} ({len(df)} rows)")

            if export_json:
                out_json = OUTPUT_DIR / f"teamdata_{safe_label}.json"
                df.to_json(out_json, orient="records", indent=2)
                logger.info(f"📑 Also exported {season_label} data to {out_json}")

            # Append summary log
            run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            summary_entry = pd.DataFrame([{
                "timestamp": run_time,
                "season": season_label,
                "rows": len(df),
                "file_csv": str(out_csv),
                "file_json": str(out_json) if export_json else None,
            }])
            try:
                if SEASON_LOG.exists():
                    summary_entry.to_csv(SEASON_LOG, mode="a", header=False, index=False)
                else:
                    summary_entry.to_csv(SEASON_LOG, index=False)
                logger.info(f"📈 Season summary appended to {SEASON_LOG}")
            except Exception as e:
                logger.warning(f"Failed to append season summary: {e}")

            results[season_label] = df
        except Exception as e:
            logger.error(f"❌ Error processing {season_label} data: {e}")

    return results


def print_latest_summary():
    """Print the latest summary entry without refetching data."""
    if not SEASON_LOG.exists():
        logger.error("No season summary log found.")
        return
    try:
        df = pd.read_csv(SEASON_LOG)
        if df.empty:
            logger.warning("Season summary log is empty.")
            return
        latest = df.tail(1).iloc[0].to_dict()
        logger.info(f"📊 Latest season summary: {latest}")
    except Exception as e:
        logger.error(f"Failed to read season summary log: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA season data from config.toml")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print the latest summary log entry without fetching new data")
    parser.add_argument("--config", type=str, default=CONFIG_PATH,
                        help="Path to config.toml file")
    parser.add_argument("--season", type=str, default=None,
                        help="Fetch only a specific season defined in config.toml")
    parser.add_argument("--export-json", action="store_true",
                        help="Also export fetched data to JSON format")
    args = parser.parse_args()

    if args.summary_only:
        print_latest_summary()
    else:
        fetch_season_data(config_path=args.config,
                          season_filter=args.season,
                          export_json=args.export_json)
