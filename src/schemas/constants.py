# ============================================================
# NBA Schema Constants
# ============================================================

# Columns for historical schedule (used in generate_historical)
HISTORICAL_SCHEDULE_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "HOME_TEAM",
    "AWAY_TEAM",
    "PTS",
    "WL",
]

# Columns for enriched schedule (used in normalize)
ENRICHED_SCHEDULE_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "HOME_TEAM",
    "AWAY_TEAM",
    "PTS",
    "WL",
    "IS_HOME",
    "REST_DAYS",
    "ROLLING_PTS",
    "WIN_STREAK",
]
