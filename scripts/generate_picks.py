# ============================================================
# File: scripts/generate_picks.py
# Purpose: Generate picks from predictions using a simple EV strategy
# ============================================================

import argparse
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pathlib import Path

from core.paths import ensure_dirs, LOGS_DIR, PICKS_SUMMARY_FILE, PICKS_BANKROLL_FILE
from core.log_config import init_global_logger
from core.exceptions import DataError, PipelineError, FileError
from core.utils import ensure_columns
from core.config import (
    BASE_RESULTS_DIR,
    PICKS_LOG,
    TODAY_PREDICTIONS_FILE,
    PICKS_FILE,
    log_config_snapshot,
)
from notifications import send_telegram_message, send_ev_summary, send_photo

logger = init_global_logger()

# Ensure results directory exists
ensure_dirs(strict=False)
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def update_bankroll(picks_df: pd.DataFrame):
    """Update bankroll tracking file with today's picks results."""
    if picks_df is None or picks_df.empty:
        return
    today = pd.Timestamp.today().date().isoformat()
    total_stake = picks_df["stake_amount"].sum() if "stake_amount" in picks_df.columns else 0
    avg_ev = picks_df["ev"].mean() if "ev" in picks_df.columns else None

    bankroll_change = 0
    if "ev" in picks_df.columns:
        if "stake_amount" in picks_df.columns:
            bankroll_change = (picks_df["ev"] * picks_df["stake_amount"]).sum()
        else:
            bankroll_change = picks_df["ev"].sum()

    record = {
        "Date": today,
        "Total_Stake": total_stake,
        "Avg_EV": avg_ev,
        "Bankroll_Change": bankroll_change,
    }

    if PICKS_BANKROLL_FILE.exists():
        hist = pd.read_csv(PICKS_BANKROLL_FILE)
        hist = pd.concat([hist, pd.DataFrame([record])], ignore_index=True)
    else:
        hist = pd.DataFrame([record])

    hist.to_csv(PICKS_BANKROLL_FILE, index=False)
    logger.info(f"💰 Bankroll updated → {PICKS_BANKROLL_FILE}")

    # Notify Telegram bankroll update
    avg_ev_str = f"{avg_ev:.3f}" if avg_ev is not None else "N/A"
    msg = (
        f"🏀 Bankroll Update ({today})\n"
        f"💰 Total Stake: {total_stake:.2f}\n"
        f"📈 Avg EV: {avg_ev_str}\n"
        f"💵 Bankroll Change: {bankroll_change:+.2f}"
    )
    send_telegram_message(msg)


def generate_picks(preds_file=TODAY_PREDICTIONS_FILE,
                   out_file=PICKS_FILE,
                   export_json: bool = False) -> pd.DataFrame:
    """Generate picks from predictions using a simple EV strategy."""
    log_config_snapshot()
    preds_file = Path(preds_file)
    out_file = Path(out_file)

    if not preds_file.exists():
        raise FileError("Predictions file not found", file_path=str(preds_file))

    df = pd.read_csv(preds_file)
    if df.empty:
        raise DataError(f"{preds_file} is empty. No predictions available.")

    # Validate required probability column
    if "win_prob" in df.columns:
        prob_col = "win_prob"
    elif "pred_home_win_prob" in df.columns:
        prob_col = "pred_home_win_prob"
    else:
        raise DataError("Predictions file missing required probability column")

    ensure_columns(df, [prob_col], "predictions")

    # Strategy: pick HOME if prob > 0.5 (and EV > 0 if available)
    if "ev" in df.columns:
        df["pick"] = df.apply(
            lambda row: "HOME" if row[prob_col] > 0.5 and row["ev"] > 0 else "AWAY",
            axis=1,
        )
    else:
        df["pick"] = df.apply(lambda row: "HOME" if row[prob_col] > 0.5 else "AWAY", axis=1)

    # Save picks
    try:
        df.to_csv(out_file, index=False)
        logger.info(f"✅ Picks saved to {out_file} | Total picks: {len(df)}")
    except Exception as e:
        raise PipelineError(f"Failed to save picks: {e}")

    if export_json:
        out_json = out_file.with_suffix(".json")
        try:
            df.to_json(out_json, orient="records", indent=2)
            logger.info(f"📑 Picks also exported to {out_json}")
        except Exception as e:
            logger.warning(f"Failed to export picks to JSON: {e}")
    else:
        out_json = None

    # Summary stats
    home_picks = (df["pick"] == "HOME").sum()
    away_picks = (df["pick"] == "AWAY").sum()
    avg_ev = df["ev"].mean() if "ev" in df.columns else None
    avg_ev_str = f"{avg_ev:.3f}" if avg_ev is not None else "N/A"

    logger.info(f"Picks summary: HOME={home_picks}, AWAY={away_picks}, Avg EV={avg_ev_str}")

    # Append to rolling log
    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_entry = pd.DataFrame([{
        "timestamp": run_time,
        "total_picks": len(df),
        "home_picks": home_picks,
        "away_picks": away_picks,
        "avg_ev": avg_ev,
        "json_export": str(out_json) if out_json else None,
    }])

    try:
        if PICKS_LOG.exists():
            summary_entry.to_csv(PICKS_LOG, mode="a", header=False, index=False)
        else:
            summary_entry.to_csv(PICKS_LOG, index=False)
        logger.info(f"📈 Picks summary appended to {PICKS_LOG}")
    except Exception as e:
        raise PipelineError(f"Failed to append picks summary: {e}")

    # ✅ Update bankroll + notify Telegram
    update_bankroll(df)
    send_ev_summary(df)
    summary_msg = f"📊 Picks Summary: HOME={home_picks}, AWAY={away_picks}, Avg EV={avg_ev_str}"
    send_telegram_message(summary_msg)

    # Append summary to dedicated log
    try:
        if PICKS_SUMMARY_FILE.exists():
            summary_entry.to_csv(PICKS_SUMMARY_FILE, mode="a", header=False, index=False)
        else:
            summary_entry.to_csv(PICKS_SUMMARY_FILE, index=False)
        logger.info(f"📈 Picks summary also appended to {PICKS_SUMMARY_FILE}")
    except Exception as e:
        logger.warning(f"Failed to append to dedicated picks summary log: {e}")

    # Optional: send bankroll trend chart
    trend_img = plot_bankroll_trend()
    if trend_img:
        send_photo(trend_img, caption="📈 Cumulative Bankroll Trend")

    return df


def plot_bankroll_trend() -> str:
    """Plot cumulative bankroll trend over time."""
    if not Path(PICKS_BANKROLL_FILE).exists():
        logger.warning("No bankroll file found.")
        return ""
    df = pd.read_csv(PICKS_BANKROLL_FILE)
    if df.empty:
        logger.warning("Bankroll file is empty.")
        return ""

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["Bankroll_Change"].cumsum(), marker="o", label="Cumulative Bankroll")
    ax.set_title("Bankroll Trend Over Time")
    ax.set_ylabel("Cumulative Change")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    trend_path = LOGS_DIR / "bankroll_trend.png"
    plt.tight_layout()
    plt.savefig(trend_path)
    plt.close()
    logger.info(f"📊 Bankroll trend saved → {trend_path}")
    return str(trend_path)


def print_latest_summary():
    """Print the latest picks summary entry without regenerating picks."""
    if not PICKS_SUMMARY_FILE.exists():
        logger.error("No picks summary log found.")
        return
    try:
        df = pd.read_csv(PICKS_SUMMARY_FILE)
        if df.empty:
            logger.warning("Picks summary log is empty.")
            return
        latest = df.tail(1).iloc[0].to_dict()
        logger.info(f"📊 Latest picks summary: {latest}")
    except Exception as e:
        logger.error(f"Failed to read picks summary log: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate picks from predictions")
    parser.add_argument("--preds", default=TODAY_PREDICTIONS_FILE, help="Path to predictions file")
    parser.add_argument("--out", default=PICKS_FILE, help="Path to save picks")
    parser.add_argument("--export-json", action="store_true", help="Also export picks to JSON format")
    args = parser.parse_args()

    generate_picks(preds_file=args.preds, out_file=args.out, export_json=args.export_json)
