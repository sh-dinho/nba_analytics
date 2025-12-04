# ============================================================
# File: scripts/plot_combined_feature_trends.py
# Purpose: Plot combined trends of training and new game feature builds
#          Includes row counts and rolling window size over time
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from core.paths import LOGS_DIR
from scripts.build_features_for_training import TRAINING_FEATURES_LOG
from scripts.build_features_for_new_games import NEW_GAMES_FEATURES_LOG


def plot_combined_feature_trends():
    """Plot combined trends of training and new game feature builds over time, including rolling window size."""
    dfs = []

    # Load training features log
    if TRAINING_FEATURES_LOG.exists():
        df_train = pd.read_csv(TRAINING_FEATURES_LOG)
        if not df_train.empty:
            df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])
            dfs.append(("Training", df_train))
    else:
        print("⚠️ No training features log found.")

    # Load new games features log
    if NEW_GAMES_FEATURES_LOG.exists():
        df_new = pd.read_csv(NEW_GAMES_FEATURES_LOG)
        if not df_new.empty:
            df_new["timestamp"] = pd.to_datetime(df_new["timestamp"])
            dfs.append(("New Games", df_new))
    else:
        print("⚠️ No new games features log found.")

    if not dfs:
        print("⚠️ No logs available for plotting.")
        return ""

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Primary axis: row counts
    for label, df in dfs:
        if label == "Training":
            ax1.plot(df["timestamp"], df["team_rows"], marker="o", label="Training Team Rows")
            ax1.plot(df["timestamp"], df["player_rows"], marker="x", label="Training Player Rows")
        elif label == "New Games":
            ax1.plot(df["timestamp"], df["rows"], marker="s", label="New Games Rows")

    ax1.set_xlabel("Run Timestamp")
    ax1.set_ylabel("Row Counts")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Secondary axis: rolling window size
    ax2 = ax1.twinx()
    for label, df in dfs:
        if "window" in df.columns:
            ax2.plot(df["timestamp"], df["window"], linestyle="--", marker="d", label=f"{label} Window Size")

    ax2.set_ylabel("Rolling Window Size")
    ax2.legend(loc="upper right")

    fig.suptitle("Combined Feature Generation Trends (Rows + Rolling Window)")
    trend_path = LOGS_DIR / "combined_features_trends.png"
    plt.tight_layout()
    plt.savefig(trend_path)
    plt.close()
    print(f"📊 Combined feature trends saved → {trend_path}")
    return str(trend_path)


if __name__ == "__main__":
    plot_combined_feature_trends()