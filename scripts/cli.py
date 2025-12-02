# File: scripts/cli.py
import argparse
import os
import sys
import pandas as pd
import logging
from datetime import datetime
from app.predict_pipeline import generate_predictions
from scripts.simulate_bankroll import simulate_bankroll
from nba_analytics_core.notifications import send_telegram_message, send_ev_summary
import json

REQUIRED_PRED_COLS = {"pred_home_win_prob", "decimal_odds", "ev"}

# ---------------------------- Logging ----------------------------
logger = logging.getLogger("cli")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)

# ---------------------------- Helpers ----------------------------
def _ensure_columns(df, required_cols, name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ---------------------------- Pipeline Runner ----------------------------
def run_pipeline(threshold, strategy, max_fraction, export, notify=False, plot=False):
    """Generate predictions, simulate bankroll, save results, optionally notify/plot."""
    try:
        df = generate_predictions(
            threshold=threshold,
            strategy=strategy,
            max_fraction=max_fraction,
            cli=True
        )
    except Exception as e:
        logger.error(f"❌ Error generating predictions: {e}")
        return None, None

    if df.empty:
        logger.info("No predictions available today.")
        return None, None

    _ensure_columns(df, REQUIRED_PRED_COLS, "predictions")

    # Simulate bankroll
    sim_df = df.rename(columns={"pred_home_win_prob": "prob"})
    history, metrics = simulate_bankroll(
        sim_df[["decimal_odds", "prob", "ev"]],
        strategy=strategy,
        max_fraction=max_fraction
    )

    # Attach bankroll history
    df["bankroll"] = pd.Series(history[1:][:len(df)])

    # Log metrics
    logger.info("=== BANKROLL METRICS ===")
    logger.info(f'Final Bankroll: ${metrics["final_bankroll"]:.2f}')
    logger.info(f'ROI: {metrics["roi"]*100:.2f}%')
    logger.info(f'Win Rate: {metrics["win_rate"]*100:.2f}% ({metrics["wins"]}W/{metrics["losses"]}L)')

    # Ensure export directory exists
    os.makedirs(os.path.dirname(export) or ".", exist_ok=True)

    # Save main CSV and timestamped backup
    df.to_csv(export, index=False)
    ts_file = export.replace(".csv", f"_{_timestamp()}.csv")
    df.to_csv(ts_file, index=False)
    logger.info(f"📂 Detailed results saved to {export}")
    logger.info(f"📦 Timestamped backup saved to {ts_file}")

    # Save summary CSV and timestamped backup
    summary_path = export.replace(".csv", "_summary.csv")
    pd.DataFrame([metrics]).to_csv(summary_path, index=False)
    ts_summary_path = summary_path.replace(".csv", f"_{_timestamp()}.csv")
    pd.DataFrame([metrics]).to_csv(ts_summary_path, index=False)
    logger.info(f"📂 Summary saved to {summary_path}")
    logger.info(f"📦 Timestamped summary backup saved to {ts_summary_path}")

    # Save metadata JSON
    meta = {
        "generated_at": datetime.now().isoformat(),
        "threshold": threshold,
        "strategy": strategy,
        "max_fraction": max_fraction,
        "rows": len(df),
        "columns": df.columns.tolist(),
        "export_file": export
    }
    meta_file = export.replace(".csv", "_meta.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"🧾 Metadata saved to {meta_file}")

    # Optional Telegram notifications
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
            logger.info("✅ Telegram notification sent")
        except Exception as e:
            logger.warning(f"⚠️ Failed to send Telegram notification: {e}")

    # Optional plot of bankroll trajectory
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
            logger.warning("⚠️ matplotlib not installed. Install it to enable plotting.")
        except Exception as e:
            logger.warning(f"⚠️ Error plotting bankroll trajectory: {e}")

    return df, metrics

# ---------------------------- Show Metrics ----------------------------
def show_metrics(export):
    summary_path = export.replace(".csv", "_summary.csv")
    if not os.path.exists(summary_path):
        logger.warning(f"❌ No summary file found at {summary_path}. Run the pipeline first.")
        return
    metrics = pd.read_csv(summary_path).iloc[0].to_dict()
    logger.info("=== LAST SAVED METRICS ===")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            # Display percentages properly
            logger.info(f"{k}: {v*100:.2f}%" if "rate" in k.lower() else f"{k}: {v:.4f}")
        else:
            logger.info(f"{k}: {v}")

# ---------------------------- CLI Entry ----------------------------
def main():
    parser = argparse.ArgumentParser(description="NBA Analytics CLI")
    parser.add_argument("--threshold", type=float, default=0.6, help="Strong pick probability threshold")
    parser.add_argument("--strategy", choices=["kelly", "flat"], default="kelly", help="Betting strategy")
    parser.add_argument("--max_fraction", type=float, default=0.05, help="Max Kelly fraction per bet")
    parser.add_argument("--export", type=str, default="results/picks.csv", help="Export CSV path")
    parser.add_argument("--notify", action="store_true", help="Send Telegram notification")
    parser.add_argument("--plot", action="store_true", help="Plot bankroll trajectory")
    parser.add_argument("--metrics", action="store_true", help="Show last saved metrics")
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
