# ============================================================
# File: scripts/data_sources.py
# Purpose: Fetch NBA games, team stats, and odds for current season
# ============================================================

import pandas as pd
from datetime import datetime, date
from core.config_loader import ConfigLoader

# Dummy fetchers (replace with real API calls)
def get_games_for_season(season_label: str) -> pd.DataFrame:
    # Replace with actual NBA API call
    return pd.DataFrame({
        "GAME_DATE": ["2025-10-21", "2025-10-22"],
        "TEAM_NAME": ["Boston Celtics", "LA Lakers"],
        "MATCHUP": ["BOS vs. LAL", "LAL @ BOS"]
    })

def get_team_stats_for_season(season_label: str) -> pd.DataFrame:
    # Replace with actual NBA API call
    return pd.DataFrame({
        "TEAM_ID": [1610612738, 1610612747],
        "TEAM_NAME": ["Boston Celtics", "LA Lakers"],
        "GP": [82, 82],
        "W": [60, 55],
        "L": [22, 27]
    })

def get_odds() -> list:
    # Replace with actual odds API call
    return [{"game": "BOS vs LAL", "odds": "+150"}]

# Logging helper
def log_event(message: str, header: bool = False):
    with open("pipeline.log", "a") as log_file:
        if header:
            log_file.write(f"\n=== {date.today().isoformat()} Run ===\n")
        log_file.write(f"[{datetime.now().isoformat()}] {message}\n")

# ---------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------
if __name__ == "__main__":
    loader = ConfigLoader("config.toml")

    # Start of run
    log_event("Pipeline run started", header=True)

    test_season = loader.ensure_current_season_blocks()
    season_data = loader.get_season("get-data", test_season)

    if loader.validate_season(season_data):
        # Games
        print(f"📅 Fetching NBA games for season {test_season}...")
        games = get_games_for_season(test_season)
        print(f"✅ {len(games)} games found.")
        games_csv = f"nba_games_{test_season.replace('-', '')}.csv"
        games.to_csv(games_csv, index=False)
        print(f"💾 Games saved to {games_csv}")
        log_event(f"Saved {len(games)} games to {games_csv} for season {test_season}")

        # Team stats
        print(f"📊 Fetching team stats for season {test_season}...")
        stats = get_team_stats_for_season(test_season)
        print(f"✅ {len(stats)} team stats rows found.")
        stats_csv = f"nba_team_stats_{test_season.replace('-', '')}.csv"
        stats.to_csv(stats_csv, index=False)
        print(f"💾 Team stats saved to {stats_csv}")
        log_event(f"Saved {len(stats)} team stats rows to {stats_csv} for season {test_season}")

                # Odds
        print(f"🎲 Fetching NBA odds...")
        odds_data = get_odds()
        if odds_data:
            print(f"✅ {len(odds_data)} odds entries found.")
            log_event(f"Fetched {len(odds_data)} odds entries for season {test_season}")
        else:
            print("❌ No odds data available.")
            log_event(f"Failed to fetch odds for season {test_season}")

        # Build API URL
        url = loader.build_data_url(season_data)
        print(f"🔗 Built API URL: {url}")
        log_event(f"Built API URL for season {test_season}: {url}")

        # End of run summary
        print("🏁 Run completed successfully.")
        log_event("Run completed successfully")
    else:
        print(f"❌ Validation failed for season {test_season}")
        log_event(f"Run failed: validation error for season {test_season}")