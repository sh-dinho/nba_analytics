# File: scripts/post_season_analysis.py
import pandas as pd
import os

def main():
    archive_dir = "season_archive"
    picks_file = os.path.join(archive_dir, "all_picks.csv")
    preds_file = os.path.join(archive_dir, "all_predictions.csv")

    if not os.path.exists(picks_file) or not os.path.exists(preds_file):
        raise FileNotFoundError("Season archive files not found. Run season_end workflow first.")

    picks = pd.read_csv(picks_file)
    preds = pd.read_csv(preds_file)

    # --- Accuracy ---
    if "actual_result" in picks.columns:
        picks["correct"] = (picks["pick"] == picks["actual_result"]).astype(int)
        accuracy = picks["correct"].mean()
        print(f"🏀 Pick Accuracy: {accuracy:.2%}")
    else:
        print("⚠️ No 'actual_result' column in picks file. Add actual outcomes to measure accuracy.")

    # --- ROI (example: betting odds if available) ---
    if "odds" in picks.columns and "actual_result" in picks.columns:
        picks["roi"] = picks.apply(
            lambda row: row["odds"] if row["pick"] == row["actual_result"] else -1, axis=1
        )
        roi = picks["roi"].mean()
        print(f"💰 Average ROI per bet: {roi:.2f}")
    else:
        print("⚠️ No odds data available for ROI calculation.")

    # --- Prediction calibration ---
    if "home_win_prob" in preds.columns and "actual_result" in preds.columns:
        preds["predicted"] = preds["home_win_prob"].apply(lambda p: "W" if p >= 0.5 else "L")
        preds["correct"] = (preds["predicted"] == preds["actual_result"]).astype(int)
        pred_acc = preds["correct"].mean()
        print(f"📊 Prediction Accuracy: {pred_acc:.2%}")
    else:
        print("⚠️ Predictions missing 'actual_result' column for calibration.")

    # --- Trend impact (example: correlation with player trends) ---
    if "points_trend" in preds.columns and "home_win_prob" in preds.columns:
        corr = preds["points_trend"].corr(preds["home_win_prob"])
        print(f"📈 Correlation between points trend and win probability: {corr:.2f}")

if __name__ == "__main__":
    main()