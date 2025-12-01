# File: scripts/generate_picks.py
import pandas as pd
import os

def main():
    os.makedirs("results", exist_ok=True)

    # Load predictions
    preds_file = "results/predictions.csv"
    if not os.path.exists(preds_file):
        raise FileNotFoundError(f"{preds_file} not found. Run predict_pipeline.py first.")

    df = pd.read_csv(preds_file)

    # Ensure required columns exist
    required_cols = {"date", "home_team", "away_team", "home_win_prob", "away_win_prob"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns in {preds_file}: {required_cols - set(df.columns)}")

    # Generate picks: choose team with higher win probability
    df["pick"] = df.apply(
        lambda row: row["home_team"] if row["home_win_prob"] >= row["away_win_prob"] else row["away_team"],
        axis=1
    )

    # Save picks file
    picks_file = "results/picks.csv"
    df[["date", "home_team", "away_team", "pick"]].to_csv(picks_file, index=False)
    print(f"✅ Picks saved to {picks_file}")

if __name__ == "__main__":
    main()