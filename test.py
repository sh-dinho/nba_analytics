import pandas as pd
from pathlib import Path
from datetime import date

odds_dir = Path("data/odds")
odds_dir.mkdir(parents=True, exist_ok=True)

today = date(2025, 12, 21)

df = pd.DataFrame(
    [
        {
            "game_id": "TEST_GAME_1",
            "team": "LAL",
            "market": "moneyline",
            "price": -110,
        },
        {
            "game_id": "TEST_GAME_1",
            "team": "BOS",
            "market": "moneyline",
            "price": +100,
        },
    ]
)

df.to_parquet(odds_dir / f"odds_{today}.parquet", index=False)
