# File: scripts/fetch_player_stats.py
import os
import time
import requests
import pandas as pd
import argparse
import sys
import logging
import json
from datetime import datetime

NBA_API_URL = "https://stats.nba.com/stats/playergamelog"
ALL_PLAYERS_URL = "https://stats.nba.com/stats/commonallplayers"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nba.com"
}

# ----------------------------
# Logging setup
# ----------------------------
logger = logging.getLogger("fetch_player_stats")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_active_players(season="2024-25") -> pd.DataFrame:
    """Fetch all active NBA players for a given season."""
    params = {
        "LeagueID": "00",
        "Season": season,
        "IsOnlyCurrentSeason": "1"
    }
    try:
        r = requests.get(ALL_PLAYERS_URL, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()["resultSets"][0]
        df = pd.DataFrame(data["rowSet"], columns=data["headers"])
        active = df[df["ROSTERSTATUS"] == 1]
        return active[["PERSON_ID", "DISPLAY_FIRST_LAST"]]
    except Exception as e:
        logger.error(f"❌ Error fetching active players: {e}")
        return pd.DataFrame(columns=["PERSON_ID", "DISPLAY_FIRST_LAST"])


def fetch_player_stats(player_id, season="2024-25", retries=3) -> pd.DataFrame | None:
    """Fetch game logs for a single player with retry/backoff."""
    params = {
        "PlayerID": player_id,
        "Season": season,
        "SeasonType": "Regular Season"
    }
    for attempt in range(retries):
        try:
            r = requests.get(NBA_API_URL, params=params, headers=HEADERS, timeout=10)
            r.raise_for_status()
            data = r.json()["resultSets"][0]
            df = pd.DataFrame(data["rowSet"], columns=data["headers"])
            df["PlayerID"] = player_id
            return df
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"⚠️ Attempt {attempt+1} failed for PlayerID {player_id}: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    logger.error(f"❌ Failed to fetch stats for PlayerID {player_id} after {retries} attempts.")
    return None


def main(season="2024-25", resume=False):
    os.makedirs("data", exist_ok=True)
    logger.info(f"Fetching active players list for season {season}...")
    active_players = get_active_players(season)
    logger.info(f"Found {len(active_players)} active players.")

    # Resume support: skip players already in saved file
    existing_file = "data/player_stats.csv"
    fetched_ids = set()
    existing_df = None
    if resume and os.path.exists(existing_file):
        existing_df = pd.read_csv(existing_file)
        fetched_ids = set(existing_df["PlayerID"].unique())
        logger.info(f"Resuming: {len(fetched_ids)} players already fetched.")

    all_stats = []
    for idx, row in active_players.iterrows():
        pid = row["PERSON_ID"]
        name = row["DISPLAY_FIRST_LAST"]
        if pid in fetched_ids:
            logger.info(f"[{idx+1}/{len(active_players)}] Skipping {name} (already fetched).")
            continue
        logger.info(f"[{idx+1}/{len(active_players)}] Fetching stats for {name} (ID {pid})...")
        df = fetch_player_stats(pid, season)
        if df is not None and not df.empty:
            all_stats.append(df)
        time.sleep(1)  # polite delay

    if all_stats:
        final_df = pd.concat(all_stats, ignore_index=True)
        if resume and existing_df is not None:
            final_df = pd.concat([existing_df, final_df], ignore_index=True)

        final_df.to_csv(existing_file, index=False)
        ts_file = f"data/player_stats_{_timestamp()}.csv"
        final_df.to_csv(ts_file, index=False)

        summary = final_df.groupby("PlayerID").size().reset_index(name="games_played")
        summary.to_csv("data/player_stats_summary.csv", index=False)

        logger.info("✅ Player stats saved to data/player_stats.csv")
        logger.info(f"📦 Timestamped backup saved to {ts_file}")
        logger.info("✅ Summary saved to data/player_stats_summary.csv")

        # Save metadata
        meta = {
            "generated_at": datetime.now().isoformat(),
            "season": season,
            "rows": len(final_df),
            "columns": final_df.columns.tolist(),
            "players_fetched": len(final_df["PlayerID"].unique()),
            "resume_mode": resume
        }
        meta_file = "data/player_stats_meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"🧾 Metadata saved to {meta_file}")
    else:
        logger.error("❌ No stats fetched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA player stats")
    parser.add_argument("--season", type=str, default="2024-25", help="Season (e.g., 2024-25)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing file")
    args = parser.parse_args()

    try:
        main(season=args.season, resume=args.resume)
    except Exception as e:
        logger.error(f"❌ Script failed: {e}")
        sys.exit(1)