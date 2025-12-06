# ============================================================
# File: scripts/generate_example_data.py
# Purpose: Generate example historical and new games CSVs
# ============================================================

import pandas as pd
from pathlib import Path
import random

DATA_DIR = Path(__file__).parent.parent / "Data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Historical Games ----------------
historical_data = [
    {"game_id": 1, "team_abbreviation": "ATL", "pts": 105, "reb": 45, "ast": 22, "game_date": "2025-11-01"},
    {"game_id": 1, "team_abbreviation": "BOS", "pts": 99, "reb": 42, "ast": 20, "game_date": "2025-11-01"},
    {"game_id": 2, "team_abbreviation": "LAL", "pts": 112, "reb": 50, "ast": 25, "game_date": "2025-11-02"},
    {"game_id": 2, "team_abbreviation": "GSW", "pts": 118, "reb": 47, "ast": 28, "game_date": "2025-11-02"},
]

historical_df = pd.DataFrame(historical_data)
historical_file = DATA_DIR / "historical_games.csv"
historical_df.to_csv(historical_file, index=False)
print(f"✅ Example historical games saved → {historical_file}")

# ---------------- New Games ----------------
teams = ["CHI", "DEN", "HOU", "SAS", "PHI", "TOR"]
new_games_data = [
    {"game_id": 3, "team_abbreviation": "CHI", "pts": 0, "reb": 0, "ast": 0, "game_date": "2025-12-06"},
    {"game_id": 3, "team_abbreviation": "DEN", "pts": 0, "reb": 0, "ast": 0, "game_date": "2025-12-06"},
]

# Generate 3 upcoming games with random stats
game_id = 4
for i in range(3):
    home, away = random.sample(teams, 2)
    new_games_data.append({
        "game_id": game_id,
        "team_abbreviation": home,
        "pts": random.randint(90, 120),
        "reb": random.randint(35, 55),
        "ast": random.randint(15, 30),
        "game_date": f"2025-12-{6+i:02d}"
    })
    new_games_data.append({
        "game_id": game_id,
        "team_abbreviation": away,
        "pts": random.randint(90, 120),
        "reb": random.randint(35, 55),
        "ast": random.randint(15, 30),
        "game_date": f"2025-12-{6+i:02d}"
    })
    game_id += 1

new_games_df = pd.DataFrame(new_games_data)
new_games_file = DATA_DIR / "new_games.csv"
new_games_df.to_csv(new_games_file, index=False)
print(f"✅ Example new games saved → {new_games_file}")