# File: scripts/cli.py
import argparse
import os
import pandas as pd
from app.predict_pipeline import generate_predictions
from scripts.simulate_bankroll import simulate_bankroll
from nba_analytics_core.notifications import send_telegram_message, send_ev_summary

def main():
    parser = argparse.ArgumentParser(description="NBA Analytics CLI")
    parser.add_argument("--threshold", type=float, default=0.6, help="Strong pick probability threshold")
    parser.add_argument("--strategy", choices=["kelly", "flat"], default="kelly", help="Betting strategy")
    parser.add_argument("--max_fraction", type=float, default=0.05, help="Max Kelly fraction per bet")
    parser.add_argument("--export", type=str, default="results/picks.csv", help="Path to export detailed results as CSV")
    parser.add_argument("--notify", action="store_true", help="Send Telegram notification after run")
    parser.add_argument("--plot", action="store_true", help="Plot bankroll trajectory (optional)")
    args = parser.parse_args()

    # Generate predictions
    try:
        df = generate_predictions(
            threshold=args.threshold,
            strategy=args.strategy,
            max_fraction=args.max_fraction,
            cli=True
        )
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        return

    if df.empty:
        print("No predictions available today.")
        return

    # Run bankroll simulation
    sim_df = df.rename(columns={"pred_home_win_prob": "prob"})
    history, metrics = simulate_bankroll(
        sim_df[["decimal_odds", "prob", "ev"]],
        strategy=args.strategy,
        max_fraction=args.max_fraction
    )

    df["bankroll"] = history[1:][:len(df)]  # align lengths

    # Print metrics
    print("\n=== BANKROLL METRICS ===")
    print(f'Final Bankroll: ${metrics["final_bankroll"]:.2f}')
    print(f'ROI: {metrics["roi"]*100:.2f}%')
    print(f'Win Rate: {metrics["win_rate"]*100:.2f}% ({metrics["wins"]}W/{metrics["losses"]}L)')

    # Export results
    os.makedirs(os.path.dirname(args.export), exist_ok=True)
    df.to_csv(args.export, index=False)
    print(f"\n📂 Detailed results exported to {args.export}")

    # Export summary
    summary_path = args.export.replace(".csv", "_summary.csv")
    pd.DataFrame([metrics]).to_csv(summary_path, index=False)
    print(f"📂 Summary exported to {summary_path}")

    # Telegram notifications
    if args.notify:
        msg = (
            f"📊 CLI Bankroll Summary\n"
            f"Final Bankroll: ${metrics['final_bankroll']:.2f}\n"
            f"ROI: {metrics['roi']*100:.2f}%\n"
            f"Win Rate: {metrics['win_rate']*100:.2f}%"
        )
        send_telegram_message(msg)
        send_ev_summary(df)
        print("✅ Telegram notification sent")

    # Optional bankroll trajectory plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plt.plot(df["bankroll"])
            plt.title("Bankroll Trajectory")
            plt.xlabel("Bet #")
            plt.ylabel("Bankroll ($)")
            plt.show()
        except ImportError:
            print("⚠️ matplotlib not installed. Install it to enable plotting.")

if __name__ == "__main__":
    main()