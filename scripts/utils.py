# -----------------------------
# CLI Wrapper
# -----------------------------
if __name__ == "__main__":
    import argparse
    import sys
    import pandas as pd
    import random
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Utility functions and bankroll simulation")
    parser.add_argument("--simulate", action="store_true",
                        help="Run a sample bankroll simulation")
    parser.add_argument("--bets", type=int, default=5,
                        help="Number of sample bets to simulate")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Initial bankroll for simulation")
    parser.add_argument("--strategy", type=str, default="kelly",
                        choices=["kelly", "fixed"],
                        help="Bet sizing strategy (kelly or fixed fraction)")
    parser.add_argument("--fraction", type=float, default=0.05,
                        help="Max fraction of bankroll to stake per bet")
    parser.add_argument("--export-csv", type=str, default=None,
                        help="Path to export simulation history as CSV")
    parser.add_argument("--export-json", type=str, default=None,
                        help="Path to export simulation history as JSON")
    parser.add_argument("--summary-file", type=str, default="results/utils_simulation_summary.csv",
                        help="Optional path to append summary results")
    parser.add_argument("--weekly-summary", action="store_true",
                    help="Generate weekly summary from utils_simulation_summary.csv")
    parser.add_argument("--monthly-summary", action="store_true",
                    help="Generate monthly summary from utils_simulation_summary.csv")
    args = parser.parse_args()

    if args.simulate:
        try:
            # Create sample bets with random probabilities and odds
            bets = []
            for _ in range(args.bets):
                prob_win = round(random.uniform(0.4, 0.7), 2)   # win probability between 40–70%
                odds = round(random.uniform(1.5, 3.0), 2)       # decimal odds between 1.5–3.0
                won = random.random() < prob_win                # outcome based on probability
                bets.append({"prob_win": prob_win, "odds": odds, "won": won})

            sim = Simulation(initial_bankroll=args.bankroll)
            sim.run(bets, strategy=args.strategy, max_fraction=args.fraction)
            summary = sim.summary()

            print("=== Simulation Results ===")
            print(f"Initial Bankroll: {args.bankroll}")
            print(f"Final Bankroll:   {summary['Final_Bankroll']:.2f}")
            print(f"Total Bets:       {summary['Total_Bets']}")
            print(f"Win Rate:         {summary['Win_Rate']:.2%}")
            print(f"Avg EV:           {summary['Avg_EV']:.3f}")
            print(f"Avg Stake:        {summary['Avg_Stake']:.2f}")

            # Export history if requested
            if args.export_csv or args.export_json:
                df = pd.DataFrame(sim.history)
                if args.export_csv:
                    df.to_csv(args.export_csv, index=False)
                    print(f"📑 Simulation history exported to CSV at {args.export_csv}")
                if args.export_json:
                    df.to_json(args.export_json, orient="records", indent=2)
                    print(f"📑 Simulation history exported to JSON at {args.export_json}")

            # Append summary to file for tracking multiple runs
            if args.summary_file:
                df_summary = pd.DataFrame([{
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "initial_bankroll": args.bankroll,
                    "final_bankroll": summary["Final_Bankroll"],
                    "total_bets": summary["Total_Bets"],
                    "win_rate": summary["Win_Rate"],
                    "avg_ev": summary["Avg_EV"],
                    "avg_stake": summary["Avg_Stake"],
                    "strategy": args.strategy,
                    "fraction": args.fraction
                }])
                if Path(args.summary_file).exists():
                    df_summary.to_csv(args.summary_file, mode="a", header=False, index=False)
                else:
                    df_summary.to_csv(args.summary_file, index=False)
                print(f"📊 Summary appended to {args.summary_file}")

        except Exception as e:
            print(f"❌ Simulation failed: {e}", file=sys.stderr)
            sys.exit(1)
            
def log_weekly_summary(summary_file: str, weekly_file: str = "results/utils_weekly_summary.csv"):
    """Aggregate bankroll changes by week across simulation runs."""
    if not Path(summary_file).exists():
        print(f"⚠️ No summary file found at {summary_file}. Skipping weekly summary.")
        return
    df = pd.read_csv(summary_file)
    if df.empty or "timestamp" not in df.columns:
        print("⚠️ Summary file empty or missing timestamp column.")
        return

    df["Date"] = pd.to_datetime(df["timestamp"])
    df["Week"] = df["Date"].dt.to_period("W").astype(str)

    weekly = df.groupby("Week").agg({
        "final_bankroll": "last",
        "total_bets": "sum",
        "win_rate": "mean",
        "avg_ev": "mean",
        "avg_stake": "mean"
    }).reset_index()

    weekly["Cumulative_Bankroll"] = weekly["final_bankroll"].cummax()
    weekly.to_csv(weekly_file, index=False)
    print(f"📑 Weekly summary exported to {weekly_file}")


def log_monthly_summary(summary_file: str, monthly_file: str = "results/utils_monthly_summary.csv"):
    """Aggregate bankroll changes by month across simulation runs."""
    if not Path(summary_file).exists():
        print(f"⚠️ No summary file found at {summary_file}. Skipping monthly summary.")
        return
    df = pd.read_csv(summary_file)
    if df.empty or "timestamp" not in df.columns:
        print("⚠️ Summary file empty or missing timestamp column.")
        return

    df["Date"] = pd.to_datetime(df["timestamp"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    monthly = df.groupby("Month").agg({
        "final_bankroll": "last",
        "total_bets": "sum",
        "win_rate": "mean",
        "avg_ev": "mean",
        "avg_stake": "mean"
    }).reset_index()

    monthly["Cumulative_Bankroll"] = monthly["final_bankroll"].cummax()
    monthly.to_csv(monthly_file, index=False)
    print(f"📑 Monthly summary exported to {monthly_file}")
