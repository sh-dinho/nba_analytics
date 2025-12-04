# ============================================================
# File: core/game_results_cache.py
# Purpose: Safe caching utility for game results with validation and archiving
# ============================================================

import pandas as pd
from core.data_cache import make_cache

EXPECTED_COLUMNS = {"game_id", "team", "score", "date"}

game_results_cache = make_cache("game_results.csv", EXPECTED_COLUMNS)

load_game_results = game_results_cache["load"]
save_game_results = game_results_cache["save"]
archive_game_results = game_results_cache["archive"]
