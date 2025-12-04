# ============================================================
# File: core/player_stats_cache.py
# Purpose: Safe caching utility for player stats with validation and archiving
# ============================================================

import pandas as pd
from core.data_cache import make_cache

# Minimal columns used downstream; nba_api provides many columns. We enforce only what's needed.
EXPECTED_COLUMNS = {"TEAM_ABBREVIATION", "PTS", "AST", "REB"}

player_stats_cache = make_cache("player_stats.csv", EXPECTED_COLUMNS)

load_player_stats = player_stats_cache["load"]
save_player_stats = player_stats_cache["save"]
archive_player_stats = player_stats_cache["archive"]
