# ============================================================
# File: scripts/generate_picks.py
# Purpose: Generate picks from predictions using a simple EV strategy
# ============================================================

import pandas as pd
import datetime
from core.log_config import setup_logger
from core.exceptions import DataError, PipelineError
from core.utils import ensure_columns
from core.config import BASE_RESULTS_DIR, PICKS_LOG, TODAY_PREDICTIONS_FILE, PICKS_FILE

logger = setup_logger("generate_picks")

# Ensure results directory exists
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main(preds_file=TODAY_PREDICTIONS_FILE, out_file=PICKS_FILE) -> pd.DataFrame:
    """
    Generate picks from predictions using a simple EV strategy.
    Adds a rolling summary log for tracking.
    """
    preds_file = Path(preds_file)
    out_file = Path(out_file)

    if not preds_file.exists():
        raise FileNotFoundError(f"{preds_file} not found.")

    df = pd.read_csv(preds_file)

    # Validate required columns
    required_cols = {"pred_home_win_prob"}
    if "win_prob" in df.columns:
        required_cols = {"win_prob"}
    try:
        ensure_columns(df, required_cols, "predictions")
    except ValueError as e:
        raise DataError(str(e))

    # Simple strategy: pick HOME if prob > 0.5, else AWAY
    prob_col = "win_prob" if "win_prob" in df.columns else "pred_home_win_prob"
    df["pick"] = df.apply(lambda row: "HOME" if row[prob_col] > 0.5 else "AWAY", axis=1)

    # Save picks
    df.to_csv(out_file, index=False)
    logger.info(f"✅ Picks saved to {out_file} | Total picks: {len(df)}")

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
        "avg_ev": avg_ev
    }])

    try:
        if PICKS_LOG.exists():
            summary_entry.to_csv(PICKS_LOG, mode="a", header=False, index=False)
        else:
            summary_entry.to_csv(PICKS_LOG, index=False)
        logger.info(f"📈 Picks summary appended to {PICKS_LOG}")
    except Exception as e:
        raise PipelineError(f"Failed to append picks summary: {e}")

    return df


if __name__ == "__main__":
    from pathlib import Path
    main()