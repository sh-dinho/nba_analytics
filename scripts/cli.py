# File: scripts/cli.py
import argparse
import os
import sys
import logging
from datetime import datetime
from typing import Optional, Tuple, Dict

import pandas as pd

from scripts.simulate_bankroll import simulate_bankroll
import scripts.setup_all as setup_all
from scripts import train_model
from app.predict_pipeline import generate_today_predictions
from nba_analytics_core.notifications import send_telegram_message, send_ev_summary

REQUIRED_PICK_COLUMNS = {"home_win_prob", "decimal_odds_home", "ev_home"}

# ----------------------------
# Logging
# ----------------------------
logger = logging.getLogger("nba_cli")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)


# ----------------------------
# Helpers
# ----------------------------
def _ensure_columns(df: pd.DataFrame, required_cols, name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _append_timestamp(path: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}_{_timestamp()}{ext}"


def run_full_pipeline() -> Optional[Dict]:
    logger.info("🚀 Running full pipeline setup...")
    try:
        setup_all.main()
        metrics = train_model.main()
        logger.info("✅ Training complete. Metrics returned to CLI.")
        return metrics
    except Exception as e:
        logger.error(f"❌ Error running full pipeline: {e}")
        return None


def load_or_generate_picks(export_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(export_path):
        logger.warning(f"❌ No picks file found at {export_path}. Generating predictions...")
        try:
            generate_today_predictions()
        except Exception as e:
            logger.error(f"❌ Error generating predictions: {e}")

    if not os.path.exists(export_path):
        logger.error("❌ Still no picks file available. Aborting.")
        return None

    df = pd.read_csv(export_path)
    if df.empty:
        logger.info("No picks available today.")
        return None

    _ensure_columns(df, REQUIRED_PICK_COLUMNS, os.path.basename(export_path))

    # Basic sanity checks
    if (df["decimal_odds_home"] <= 1).any():
        logger.warning("Detected non-sensical decimal odds (<=1). Some rows may be skipped downstream.")

    return df


def simulate(
    df: pd.DataFrame,
    strategy: str,
    max_fraction: float,
    export_path: str,
    timestamp_files: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    try:
        sim_df = df.rename(columns={"home_win_prob": "prob"})
        bankroll_input = sim_df[["decimal_odds_home", "prob", "ev_home"]].rename(
            columns={"decimal_odds_home": "decimal_odds", "ev_home": "ev"}
        )

        history, metrics = simulate_bankroll(
            bankroll_input,
            strategy=strategy,
            max_fraction=max_fraction
        )

        # Attach bankroll trajectory aligned to bets
        df = df.copy()
        df["bankroll"] = history[:len(df)]

        logger.info("\n=== BANKROLL METRICS ===")
        logger.info(f'Final Bankroll: ${metrics["final_bankroll"]:.2f}')
        logger.info(f'ROI: {metrics["roi"]*100:.2f}%')
        logger.info(f'Win Rate: {metrics["win_rate"]*100:.2f}% ({metrics["wins"]}W/{metrics["losses"]}L)')

        # Export results
        os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
        export_detailed = _append_timestamp(export_path) if timestamp_files else export_path
        df.to_csv(export_detailed, index=False)
        logger.info(f"📂 Detailed results exported to {export_detailed}")

        export_summary = export_path.replace(".csv", "_summary.csv")
        export_summary = _append_timestamp(export_summary) if timestamp_files else export_summary
        pd.DataFrame([metrics]).to_csv(export_summary, index=False)
        logger.info(f"📂 Summary exported to {export_summary}")

        # Simple risk-adjusted metric
        if "roi" in metrics and "roi_std" not in metrics:
            # Recompute ROI std from incremental returns if available later
            pass

        return df, metrics
    except Exception as e:
        logger.error(f"❌ Error running bankroll simulation: {e}")
        return None, None


def notify(df: pd.DataFrame, metrics: Dict) -> None:
    try:
        msg = (
            f"📊 CLI Bankroll Summary\n"
            f"Final Bankroll: ${metrics['final_bankroll']:.2f}\n"
            f"ROI: {metrics['roi']*100:.2f}%\n"
            f"Win Rate: {metrics['win_rate']*100:.2f}%"
        )
        send_telegram_message(msg)
        # Send EV breakdown for the day (relies on df having EV columns)
        send_ev_summary(df)
        logger.info("✅ Telegram notification sent")
    except Exception as e:
        logger.warning(f"⚠️ Failed to send Telegram notification: {e}")


def plot(df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(df["bankroll"], marker="o")
        plt.title("Bankroll trajectory")
        plt.xlabel("Bet #")
        plt.ylabel("Bankroll ($)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        logger.warning("⚠️ matplotlib not installed. Install it to enable plotting.")
    except Exception as e:
        logger.warning(f"⚠️ Error plotting bankroll trajectory: {e}")


def show_metrics(export_path: str) -> None:
    summary_path = export_path.replace(".csv", "_summary.csv")
    # Pick latest timestamped summary if exists
    base, ext = os.path.splitext(summary_path)
    directory = os.path.dirname(summary_path) or "."
    candidates = [p for p in os.listdir(directory) if os.path.basename(base) in p and p.endswith(ext)]
    if candidates:
        # Choose lexicographically last (latest timestamp format ensures correct ordering)
        summary_file = os.path.join(directory, sorted(candidates)[-1])
    else:
        summary_file = summary_path

    if not os.path.exists(summary_file):
        logger.error(f"❌ No summary file found at {summary_file}. Run a simulation first.")
        return

    metrics = pd.read_csv(summary_file).iloc[0].to_dict()
    logger.info("\n=== LAST SAVED METRICS ===")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            if k.endswith("_rate"):
                logger.info(f"{k}: {v*100:.2f}%")
            else:
                logger.info(f"{k}: {v:.4f}")
        else:
            logger.info(f"{k}: {v}")


# ----------------------------
# CLI entry
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="NBA Analytics CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Full pipeline (fetch, train, predict, picks)
    full_parser = subparsers.add_parser("full", help="Run full pipeline (fetch, train, predict, picks)")
    full_parser.add_argument("--picks", type=str, default="results/picks.csv", help="Path to picks CSV to check/create")

    # Simulation
    sim_parser = subparsers.add_parser("simulate", help="Run bankroll simulation")
    sim_parser.add_argument("--strategy", choices=["kelly", "flat"], default="kelly")
    sim_parser.add_argument("--max_fraction", type=float, default=0.05)
    sim_parser.add_argument("--export", type=str, default="results/picks.csv")
    sim_parser.add_argument("--no-ts", action="store_true", help="Do not timestamp output filenames")

    # Notification
    notify_parser = subparsers.add_parser("notify", help="Send Telegram notification")
    notify_parser.add_argument("--export", type=str, default="results/picks.csv")

    # Plot
    plot_parser = subparsers.add_parser("plot", help="Plot bankroll trajectory")
    plot_parser.add_argument("--export", type=str, default="results/picks.csv")

    # Quick run: simulate, notify, and plot
    run_parser = subparsers.add_parser("run", help="Quick run: simulate, notify, and plot")
    run_parser.add_argument("--strategy", choices=["kelly", "flat"], default="kelly")
    run_parser.add_argument("--max_fraction", type=float, default=0.05)
    run_parser.add_argument("--export", type=str, default="results/picks.csv")
    run_parser.add_argument("--no-ts", action="store_true", help="Do not timestamp output filenames")

    # Show last metrics
    metrics_parser = subparsers.add_parser("metrics", help="Show last saved bankroll metrics")
    metrics_parser.add_argument("--export", type=str, default="results/picks.csv")

    args = parser.parse_args()

    if args.command == "full":
        run_full_pipeline()
        # Optionally generate predictions into args.picks if not present
        if not os.path.exists(args.picks):
            try:
                generate_today_predictions()
            except Exception as e:
                logger.error(f"❌ Error generating predictions in full pipeline: {e}")

    elif args.command == "simulate":
        df = load_or_generate_picks(args.export)
        if df is not None:
            simulate(df, args.strategy, args.max_fraction, args.export, timestamp_files=not args.no_ts)

    elif args.command == "notify":
        df = load_or_generate_picks(args.export)
        if df is not None:
            # Prefer to load latest summary if available rather than re-simulating with default params
            summary_path = args.export.replace(".csv", "_summary.csv")
            try:
                # Best effort: read last metrics (timestamp-aware)
                base, ext = os.path.splitext(summary_path)
                directory = os.path.dirname(summary_path) or "."
                candidates = [p for p in os.listdir(directory) if os.path.basename(base) in p and p.endswith(ext)]
                metrics = None
                if candidates:
                    summary_file = os.path.join(directory, sorted(candidates)[-1])
                    metrics = pd.read_csv(summary_file).iloc[0].to_dict()
                if metrics is None:
                    # Fallback: run a quick simulation with defaults
                    _, metrics = simulate(df, "kelly", 0.05, args.export, timestamp_files=False)
                if metrics:
                    notify(df, metrics)
            except Exception as e:
                logger.warning(f"⚠️ Failed to prepare metrics for notification: {e}")

    elif args.command == "plot":
        df = load_or_generate_picks(args.export)
        if df is not None:
            plot(df)

    elif args.command == "run":
        df = load_or_generate_picks(args.export)
        if df is not None:
            df, metrics = simulate(df, args.strategy, args.max_fraction, args.export, timestamp_files=not args.no_ts)
            if metrics:
                notify(df, metrics)
                plot(df)

    elif args.command == "metrics":
        show_metrics(args.export)


if __name__ == "__main__":
    main()