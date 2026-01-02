import pandas as pd
from src.config.paths import LONG_SNAPSHOT

# ------------------------------------------------------------
# 1. Load predictions file for the day
# ------------------------------------------------------------
pred_path = r"C:\Users\Mohamadou\PycharmProjects\nba_analytics\data\predictions\predictions_2026-01-02.parquet"
df_pred = pd.read_parquet(pred_path)

# ------------------------------------------------------------
# 2. Load the long snapshot (contains game_date)
# ------------------------------------------------------------
df_long = pd.read_parquet(LONG_SNAPSHOT)

# ------------------------------------------------------------
# 3. Keep only the columns needed for merging
# ------------------------------------------------------------
# Your long snapshot definitely has game_id, team, and a date column.
# Let's detect the date column automatically.
date_cols = [c for c in df_long.columns if "date" in c.lower()]

if not date_cols:
    raise ValueError("No date column found in LONG_SNAPSHOT")

date_col = date_cols[0]   # e.g., 'game_date' or 'date'

df_long_small = df_long[["game_id", "team", date_col]]

# ------------------------------------------------------------
# 4. Merge predictions with dates
# ------------------------------------------------------------
merged = df_pred.merge(df_long_small, on=["game_id", "team"], how="left")

# ------------------------------------------------------------
# 5. Filter for 2026‑01‑02
# ------------------------------------------------------------
today_preds = merged[merged[date_col] == "2025-12-30"]

print(today_preds)