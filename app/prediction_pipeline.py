# ============================================================
# File: app/prediction_pipeline.py
# Purpose: Run prediction models and bankroll simulation
# ============================================================

import argparse
import pandas as pd
from scripts.utils import Simulation   # <-- NEW
from scripts.sbr_odds_provider import SbrOddsProvider

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--strategy", type=str, default="kelly")
    parser.add_argument("--max_fraction", type=float, default=0.05)
    parser.add_argument("--use_synthetic", action="store_true")
    args = parser.parse_args()

    # 1️⃣ Load data (synthetic or real odds)
    if args.use_synthetic:
        bets = [
            {"prob_win": 0.55, "odds": 2.0},
            {"prob_win": 0.65, "odds": 1.8},
            {"prob_win": 0.45, "odds": 2.2},
        ]
    else:
        provider = SbrOddsProvider()
        odds_data = provider.get_odds()
        # Convert odds_data into bets list
        bets = []
        for matchup, teams in odds_data.items():
            for team, data in teams.items():
                if team != "under_over_odds":
                    bets.append({
                        "prob_win": 0.55,  # placeholder until model predicts
                        "odds": (data["money_line_odds"] / 100) + 1 if data["money_line_odds"] > 0 else (100 / abs(data["money_line_odds"])) + 1
                    })

    # 2️⃣ Run simulation
    sim = Simulation(initial_bankroll=1000)
    sim.run(bets, strategy=args.strategy, max_fraction=args.max_fraction)

    # 3️⃣ Save results
    df = pd.DataFrame(sim.history)
    df.to_csv("results/picks_bankroll.csv", index=False)

    # 4️⃣ Print summary
    summary = sim.summary()
    print("📊 Daily Summary:", summary)


if __name__ == "__main__":
    main()