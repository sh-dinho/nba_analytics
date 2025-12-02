# File: scripts/generate_picks.py
import os
import sys
import pandas as pd
import argparse
import logging
import json
from datetime import datetime

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


def _ensure_columns(df, required_cols, name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def payout(moneyline):
    """Convert American moneyline to decimal payout (profit per $1 stake)."""
    if pd.isna(moneyline):
        return None
    try:
        ml = float(moneyline)
        return (ml / 100.0) if ml > 0 else (100.0 / abs(ml))
    except Exception:
        return None


def implied_prob(moneyline):
    """Convert American moneyline to implied probability."""
    if pd.isna(moneyline):
        return None
    try:
        ml = float(moneyline)
        if ml < 0:
            return -ml / (-ml + 100)
        else:
            return 100 / (ml + 100)
    except Exception:
        return None


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def main(preds_file="results/predictions.csv", odds_file="data/odds.csv", out_file="results/picks.csv"):
    if not os.path.exists(preds_file):
        raise FileNotFoundError("❌ Predictions file not found. Run prediction pipeline first.")

    preds = pd.read_csv(preds_file)
    _ensure_columns(preds, REQUIRED_PRED_COLS, "predictions.csv")

    odds = None
    if os.path.exists(odds_file):
        odds = pd.read_csv(odds_file)
        _ensure_columns(odds, REQUIRED_ODDS_COLS, "odds.csv")

    picks = preds.copy()

    # Baseline pick: HOME if win prob ≥ 0.5, else AWAY
    picks["pick"] = picks["home_win_prob"].apply(lambda p: "HOME" if p >= 0.5 else "AWAY")

    if odds is not None:
        picks = picks.merge(
            odds[["game_id", "home_moneyline", "away_moneyline", "spread", "total"]],
            on="game_id", how="left"
        )

        # EV calculations (safe handling of NaNs)
        picks["home_ev"] = picks.apply(
            lambda r: r["home_win_prob"] * (payout(r["home_moneyline"]) or 0) - (1 - r["home_win_prob"]),
            axis=1
        )
        picks["away_ev"] = picks.apply(
            lambda r: (1 - r["home_win_prob"]) * (payout(r["away_moneyline"]) or 0) - r["home_win_prob"],
            axis=1
        )

        picks["pick_ev_side"] = picks.apply(
            lambda r: "HOME" if r["home_ev"] >= r["away_ev"] else "AWAY",
            axis=1
        )
        picks["pick"] = picks["pick_ev_side"]

        # Add implied probabilities for transparency
        picks["home_implied_prob"] = picks["home_moneyline"].apply(implied_prob)
        picks["away_implied_prob"] = picks["away_moneyline"].apply(implied_prob)

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    picks.to_csv(out_file, index=False)

    # Save timestamped backup
    ts_file = out_file.replace(".csv", f"_{_timestamp()}.csv")
    picks.to_csv(ts_file, index=False)

    # Save summary
    summary_file = out_file.replace(".csv", "_summary.csv")
    summary = picks["pick"].value_counts().reset_index()
    summary.columns = ["side", "count"]
    summary.to_csv(summary_file, index=False)

    logger.info(f"✅ Picks saved to {out_file}")
    logger.info(f"📦 Timestamped backup saved to {ts_file}")
    logger.info(f"📊 Summary saved to {summary_file}")

    home_count = summary.loc[summary["side"] == "HOME", "count"].sum()
    away_count = summary.loc[summary["side"] == "AWAY", "count"].sum()
    logger.info(f"🔎 HOME picks: {home_count}")
    logger.info(f"🔎 AWAY picks: {away_count}")

    # Save metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "rows": len(picks),
        "columns": picks.columns.tolist(),
        "preds_file": preds_file,
        "odds_file": odds_file,
        "out_file": out_file,
        "home_count": int(home_count),
        "away_count": int(away_count)
    }
    meta_file = out_file.replace(".csv", "_meta.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"🧾 Metadata saved to {meta_file}")

    return picks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate betting picks from predictions and odds")
    parser.add_argument("--preds", type=str, default="results/predictions.csv", help="Path to predictions file")
    parser.add_argument("--odds", type=str, default="data/odds.csv", help="Path to odds file")
    parser.add_argument("--export", type=str, default="results/picks.csv", help="Path to export picks file")
    args = parser.parse_args()

    try:
        main(preds_file=args.preds, odds_file=args.odds, out_file=args.export)
    except Exception as e:
        logger.error(f"❌ Picks generation failed: {e}")
        sys.exit(1)