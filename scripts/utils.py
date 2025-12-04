# ============================================================
# File: scripts/utils.py
# Purpose: Shared utility functions and bankroll simulation
# ============================================================

# ... [all your existing functions/classes remain unchanged above] ...

# -----------------------------
# CLI Wrapper
# -----------------------------
if __name__ == "__main__":
    import argparse

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
    args = parser.parse_args()

    if args.simulate:
        # Create sample bets with random probabilities and odds
        import random
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
