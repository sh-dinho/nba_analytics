import pandas as pd
from scripts.utils import setup_logger, safe_mkdir

logger = setup_logger("build_features")

def main(player_stats_file="data/player_stats.csv", out_file="data/training_features.csv"):
    logger.info("Building features for model training")

    df = pd.read_csv(player_stats_file)

    # Example features: normalized points, assists, rebounds
    df["points_norm"] = df["points"] / df["points"].max()
    df["assists_norm"] = df["assists"] / df["assists"].max()
    df["rebounds_norm"] = df["rebounds"] / df["rebounds"].max()
    df["target_home_win"] = [1,0,1]  # Dummy target, replace with real game results

    safe_mkdir("data")
    df.to_csv(out_file, index=False)
    logger.info(f"Features saved to {out_file}")
    return df
