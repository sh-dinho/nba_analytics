import pandas as pd
from datetime import date
from src.config.paths import COMBINED_PRED_DIR

pred_date = date(2025, 12, 26)
df = pd.read_parquet(COMBINED_PRED_DIR / f"combined_{pred_date}.parquet")

winners = df.assign(
    predicted_winner=df.apply(
        lambda r: (
            r["home_team"] if r["win_probability_home"] >= 0.5 else r["away_team"]
        ),
        axis=1,
    )
)[
    [
        "game_id",
        "home_team",
        "away_team",
        "win_probability_home",
        "win_probability_away",
        "predicted_winner",
    ]
]

print(winners)
