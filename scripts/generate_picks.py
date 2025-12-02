# File: scripts/generate_picks.py

import os
import sys
import pandas as pd
import argparse
import logging
import json
from datetime import datetime
from nba_analytics_core.notifications import send_telegram_message

REQUIRED_PRED_COLS = {"game_id", "home_win_prob"}
REQUIRED_ODDS_COLS = {"game_id", "home_moneyline", "away_moneyline", "spread", "total"}

# ----------------------------
# Logging setup
# ----------------------------
logger = logging.getLogger("generate_picks")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)

# ----------------------------
# Utility functions
# ----------------------------
def _ensure_columns(df, required_cols, name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

def payout(ml):
    """Vectorized: Convert American moneyline to decimal payout per $1 stake."""
    ml = pd.to_numeric(ml, errors="coerce")
    return ml.apply(lambda x: (x / 100) if x > 0 else (100 / abs(x)) if x < 0 else 0)

def implied_prob(ml):
    """Vectorized: Convert American moneyline to implied probability."""
    ml = pd.to_numeric(ml, errors="coerce")
    return ml.apply(lambda x: -x / (-x + 100) if x < 0 else 100 / (x + 100) if x > 0 else 0)

def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ----------------------------
# Main function
# ----------------------------
def main(preds_file="results/predictions.csv",
         odds_file="data/odds.csv",
         out_file="results/picks.csv",
         notify=False):

    if not os.path.exists(preds_file):
        raise FileNotFoundError("❌ Predictions file not found. Run prediction pipeline first.")

    # Load predictions
    preds = pd.read_csv(preds_file)
    _ensure_columns(preds, REQUIRED_PRED_COLS, "predictions.csv")

    # Load odds if available
    odds = pd.DataFrame()
    if os.path.exists(odds_file):
        odds = pd.read_csv(odds_file)
        _ensure_columns(odds, REQUIRED_ODDS_COLS, "odds.csv")

    picks = preds.copy()

    # Baseline pick: HOME if win_prob >= 0.5
    picks["pick"] = picks["home_win_prob"].apply(lambda p: "HOME" if p >= 0.5 else "AWAY")
    picks["confidence"] = picks["home_win_prob"].apply(lambda p: "Strong" if p >= 0.6 else "Weak")

    if not odds.empty:
        # Merge odds
        picks = picks.merge(
            odds[["game_id", "home_moneyline", "away_moneyline", "spread", "total"]],
            on="game_id", how="left"
        )

        # Vectorized EV computation
        home_prob = picks["home_win_prob"].fillna(0)
        away_prob = 1 - home_prob
        picks["home_ev"] = home_prob * payout(picks["home_moneyline"]) - away_prob
        picks["away_ev"] = away_prob * payout(picks["away_moneyline"]) - home_prob

        # Pick based on EV
        picks["pick_ev_side"] = picks[["home_ev", "away_ev"]].idxmax(axis=1).str.replace("_ev", "").str.upper()
        # Override weak picks with EV-based pick
        picks["pick"] = picks.apply(
            lambda r: r["pick"] if r["confidence"] == "Strong" else r["pick_ev_side"], axis=1
        )

        # Add implied probabilities
        picks["home_implied_prob"] = implied_prob(picks["home_moneyline"])
        picks["away_implied_prob"] = implied_prob(picks["away_moneyline"])

    # ----------------------------
    # Save outputs
    # ----------------------------
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    picks.to_csv(out_file, index=False)
    ts_file = out_file.replace(".csv", f"_{_timestamp()}.csv")
    picks.to_csv(ts_file, index=False)

    # Save summary
    summary_file = out_file.replace(".csv", "_summary.csv")
    summary = picks["pick"].value_counts().reset_index()
    summary.columns = ["side", "count"]
    summary.to_csv(summary_file, index=False)

    # Counts
    home_count = summary.loc[summary["side"] == "HOME", "count"].sum()
    away_count = summary.loc[summary["side"] == "AWAY", "count"].sum()

    # Save metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "rows": len(picks),
        "columns": picks.columns.tolist(),
        "preds_file": preds_file,
        "odds_file": odds_file,
        "out_file": out_file,
        "home_count": int(home_count),
        "away_count": int(away_count),
        "confidence_distribution": picks["confidence"].value_counts().to_dict()
    }
    meta_file = out_file.replace(".csv", "_meta.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"✅ Picks saved to {out_file}")
    logger.info(f"📦 Timestamped backup saved to {ts_file}")
    logger.info(f"📊 Summary saved to {summary_file}")
    logger.info(f"🔎 HOME picks: {home_count}, AWAY picks: {away_count}")
    logger.info(f"🧾 Metadata saved to {meta_file}")

    # Telegram notification
    if notify and "home_ev" in picks.columns:
        top_game = picks.loc[picks[["home_ev", "away_ev"]].max(axis=1).idxmax()]
        msg = (
            f"🏀 Top EV Pick\n"
            f"{top_game.get('home_team','HOME')} vs {top_game.get('away_team','AWAY')}\n"
            f"Pick: {top_game['pick']} | EV Home: {top_game['home_ev']:.2f}, EV Away: {top_game['away_ev']:.2f}"
        )
        try:
            send_telegram_message(msg)
            logger.info("✅ Telegram notification sent for top EV pick")
        except Exception as e:
            logger.warning(f"⚠️ Failed to send Telegram message: {e}")

    return picks

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate betting picks from predictions and odds")
    parser.add_argument("--preds", type=str, default="results/predictions.csv", help="Path to predictions file")
    parser.add_argument("--odds", type=str, default="data/odds.csv", help="Path to odds file")
    parser.add_argument("--export", type=str, default="results/picks.csv", help="Path to export picks file")
    parser.add_argument("--notify", action="store_true", help="Send Telegram notification for top EV pick")
    args = parser.parse_args()

    try:
        main(preds_file=args.preds, odds_file=args.odds, out_file=args.export, notify=args.notify)
    except Exception as e:
        logger.error(f"❌ Picks generation failed: {e}")
        sys.exit(1)
