# ============================================================
# File: core/training_cache.py
# Purpose: Safe caching utility for training features with validation and archiving
# ============================================================

import pandas as pd
from core.data_cache import make_cache

EXPECTED_COLUMNS = {"game_id", "home_team", "away_team", "home_win"}

training_cache = make_cache("training_features.csv", EXPECTED_COLUMNS)

load_training = training_cache["load"]
save_training = training_cache["save"]
archive_training = training_cache["archive"]
