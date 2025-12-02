# File: scripts/fetch_player_stats_parallel.py
import os
import time
import json
import requests
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

NBA_API_URL = "https://stats.nba.com/stats/playergamelog"
ALL_PLAYERS_URL = "https://stats.nba.com/stats/commonallplayers"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nba.com"
}

logger = logging.getLogger("fetch_player_stats_parallel")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)

FAILED_FILE = "data/failed_players.csv"

def _timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_active_players(season="2024-25") -> pd.DataFrame:
    params = {"LeagueID": "00", "Season": season, "IsOnlyCurrentSeason": "1"}
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

def fetch_player(player_id, retries=3, season="2024-25"):
    params = {"PlayerID": player_id, "Season": season, "SeasonType": "Regular Season"}
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
            logger.warning(f"Attempt {attempt+1} failed for PlayerID {player_id}: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    logger.error(f"Failed to fetch stats for PlayerID {player_id} after {retries} attempts.")
    return None

def fetch_and_save_player_stats_parallel(season="2024-25", resume=True, max_workers=8):
    os.makedirs("data", exist_ok=True)
    existing_file = "data/player_stats.csv"
    existing_ids = set()
    if resume and os.path.exists(existing_file):
        df_existing = pd.read_csv(existing_file)
        existing_ids = set(df_existing["PlayerID"].unique())

    # Load failed players
    failed_players = []
    if os.path.exists(FAILED_FILE):
        failed_players = pd.read_csv(FAILED_FILE)["PlayerID"].tolist()

    active_players = get_active_players(season)
    all_players = pd.concat([
        active_players[~active_players["PERSON_ID"].isin(existing_ids)],
        pd.DataFrame({"PERSON_ID": failed_players})
    ], ignore_index=True).drop_duplicates("PERSON_ID")

    logger.info(f"Fetching stats for {len(all_players)} players (including failed retries).")

    results = []
    new_failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pid = {executor.submit(fetch_player, pid): pid for pid in all_players["PERSON_ID"]}
        for future in as_completed(future_to_pid):
            pid = future_to_pid[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    results.append(df)
                else:
                    new_failed.append(pid)
            except Exception as e:
                logger.error(f"Exception fetching PlayerID {pid}: {e}")
                new_failed.append(pid)

    if results:
        final_df = pd.concat(results, ignore_index=True)
        if resume and os.path.exists(existing_file):
            df_existing = pd.read_csv(existing_file)
            final_df = pd.concat([df_existing, final_df], ignore_index=True)

        final_df.to_csv(existing_file, index=False)
        ts_file = f"data/player_stats_{_timestamp()}.csv"
        final_df.to_csv(ts_file, index=False)
        logger.info(f"✅ Player stats saved. Backup: {ts_file}")

        # Save summary
        summary = final_df.groupby("PlayerID").size().reset_index(name="games_played")
        summary.to_csv("data/player_stats_summary.csv", index=False)

        # Save metadata
        meta = {
            "generated_at": datetime.now().isoformat(),
            "season": season,
            "rows": len(final_df),
            "columns": final_df.columns.tolist(),
            "players_fetched": len(final_df["PlayerID"].unique())
        }
        with open("data/player_stats_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"🧾 Metadata saved.")

    # Update failed_players.csv
    if new_failed:
        pd.DataFrame({"PlayerID": new_failed}).to_csv(FAILED_FILE, index=False)
        logger.warning(f"{len(new_failed)} players failed and will be retried next run.")
    elif os.path.exists(FAILED_FILE):
        os.remove(FAILED_FILE)
        logger.info("All previously failed players succeeded. Removed failed_players.csv.")
