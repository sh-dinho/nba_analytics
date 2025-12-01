import argparse
import os
import pandas as pd
from app.predict_pipeline import generate_predictions
from scripts.simulate_bankroll import simulate_bankroll

def main():
    parser = argparse.ArgumentParser(description="NBA Analytics CLI")
    parser.add_argument("--threshold", type=float, default=0.6, help="Strong pick probability threshold")
    parser.add_argument("--strategy", choices=["kelly", "flat"], default="kelly", help="Betting strategy")
    parser.add_argument("--max_fraction", type=float, default=0.05, help="Max Kelly fraction per bet")
    parser.add_argument("--export", type=str, default="results/picks.csv", help="Path to export results as CSV")
    args = parser.parse_args()

    df = generate_predictions(threshold=args.threshold, strategy=args.strategy, max_fraction=args.max_fraction, cli=True)

    if df.empty:
        print("No predictions available today.")
        return

    # bankroll simulation uses prob + odds
    sim_df = df.rename(columns={"pred_home_win_prob": "prob"})
    history, metrics = simulate_bankroll(sim_df[["decimal_odds", "prob", "ev"]], strategy=args.strategy, max_fraction=args.max_fraction)

    df["bankroll"] = history[1:][:len(df)]  # align lengths

    print("\n=== BANKROLL METRICS ===")
    print(f'Final Bankroll: ${metrics["final_bankroll"]:.2f}')
    print(f'ROI: {metrics["roi"]*100:.2f}%')
    print(f'Win Rate: {metrics["win_rate"]*100:.2f}% ({metrics["wins"]}W/{metrics["losses"]}L)')

    os.makedirs(os.path.dirname(args.export), exist_ok=True)
    df.to_csv(args.export, index=False)
    print(f"\n📂 Results exported to {args.export}")

if __name__ == "__main__":
    main()