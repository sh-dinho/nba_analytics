# File: scripts/cli.py
import argparse
import os
import pandas as pd
from app.predict_pipeline import generate_predictions
from scripts.simulate_bankroll import simulate_bankroll
from nba_analytics_core.notifications import send_telegram_message, send_ev_summary

REQUIRED_PRED_COLS = {"pred_home_win_prob", "decimal_odds", "ev"}

def _ensure_columns(df, required_cols, name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

def run_pipeline(threshold, strategy, max_fraction, export, notify=False, plot=False):
    # Generate predictions
    try:
        df = generate_predictions(
            threshold=threshold,
            strategy=strategy,
            max_fraction=max_fraction,
            cli=True
        )
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        return None, None

    if df.empty:
        print("No predictions available today.")
        return None, None

    # Validate columns
    _ensure_columns(df, REQUIRED_PRED_COLS, "predictions")

    # Run bankroll simulation
    sim_df = df.rename(columns={"pred_home_win_prob": "prob"})
    history, metrics = simulate_bankroll(
        sim_df[["decimal_odds", "prob", "ev"]],
        strategy=strategy,
        max_fraction=max_fraction
    )

    # Align bankroll history length
    df["bankroll"] = history[1:][:len(df)]

    # Print metrics
    print("\n=== BANKROLL METRICS ===")
    print(f'Final Bankroll: ${metrics["final_bankroll"]:.2f}')
    print(f'ROI: {metrics["roi"]*100:.2f}%')
    print(f'Win Rate: {metrics["win_rate"]*100:.2f}% ({metrics["wins"]}W/{metrics["losses"]}L)')

    # Export results
    os.makedirs(os.path.dirname(export), exist_ok=True)
    df.to_csv(export, index=False)
    print(f"\n📂 Detailed results exported to {export}")

    # Export summary
    summary_path = export.replace(".csv", "_summary.csv")
    pd.DataFrame([metrics]).to_csv(summary_path, index=False)
    print(f"📂 Summary exported to {summary_path}")

    # Telegram notifications
    if notify:
        msg = (
            f"📊 CLI Bankroll Summary\n"
            f"Final Bankroll: ${metrics['final_bankroll']:.2f}\n"
            f"ROI: {metrics['roi']*100:.2f}%\n"
            f"Win Rate: {metrics['win_rate']*100:.2f}%"
        )
        try:
            send_telegram_message(msg)
            send_ev_summary(df)
            print("✅ Telegram notification sent")
        except Exception as e:
            print(f"⚠️ Failed to send Telegram notification: {e}")

    # Optional bankroll trajectory plot
    if plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,4))
            plt.plot(df["bankroll"], marker="o")
            plt.title("Bankroll Trajectory")
            plt.xlabel("Bet #")
            plt.ylabel("Bankroll ($)")
            plt.grid(True)
            plt.show()
        except ImportError:
            print("⚠️ matplotlib not installed. Install it to enable plotting.")
        except Exception as e:
            print(f"⚠️ Error plotting bankroll trajectory: {e}")

    return df, metrics

def show_metrics(export):
    summary_path = export.replace(".csv", "_summary.csv")
    if not os.path.exists(summary_path):
        print(f"❌ No summary file found at {summary_path}. Run the pipeline first.")
        return
    metrics = pd.read_csv(summary_path).iloc[0].to_dict()
    print("\n=== LAST SAVED METRICS ===")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}" if not k.endswith("_rate") else f"{k}: {v*100:.2f}%")
        else:
            print(f"{k}: {v}")

def main():
    parser = argparse.ArgumentParser(description="NBA Analytics CLI")
    parser.add_argument("--threshold", type=float, default=0.6, help="Strong pick probability threshold")
    parser.add_argument("--strategy", choices=["kelly", "flat"], default="kelly", help="Betting strategy")
    parser.add_argument("--max_fraction", type=float, default=0.05, help="Max Kelly fraction per bet")
    parser.add_argument("--export", type=str, default="results/picks.csv", help="Path to export detailed results as CSV")
    parser.add_argument("--notify", action="store_true", help="Send Telegram notification after run")
    parser.add_argument("--plot", action="store_true", help="Plot bankroll trajectory (optional)")
    parser.add_argument("--metrics", action="store_true", help="Show last saved bankroll metrics without rerunning pipeline")
    args = parser.parse_args()

    if args.metrics:
        show_metrics(args.export)
    else:
        run_pipeline(
            threshold=args.threshold,
            strategy=args.strategy,
            max_fraction=args.max_fraction,
            export=args.export,
            notify=args.notify,
            plot=args.plot
        )

if __name__ == "__main__":
    main()