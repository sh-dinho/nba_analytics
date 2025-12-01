# File: scripts/build_training_data.py
import pandas as pd
import os

def main():
    os.makedirs("data", exist_ok=True)

    # Load weekly summary and player trends
    weekly_file = "results/weekly_summary.csv"
    trends_file = "results/player_trends.csv"

    if not os.path.exists(weekly_file) or not os.path.exists(trends_file):
        raise FileNotFoundError("Weekly summary or player trends file not found. Run those scripts first.")

    weekly = pd.read_csv(weekly_file)
    trends = pd.read_csv(trends_file)

    # Example merge: join on team and date
    df = weekly.merge(trends, on=["team", "date"], how="left")

    # Add target column (e.g., win/loss from weekly summary)
    if "result" in weekly.columns:
        df["target"] = (weekly["result"] == "W").astype(int)
    else:
        raise ValueError("Weekly summary must contain a 'result' column with W/L values.")

    # Save training dataset
    out_file = "data/training_data.csv"
    df.to_csv(out_file, index=False)
    print(f"✅ Training dataset built and saved to {out_file}")

if __name__ == "__main__":
    main()