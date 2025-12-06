# ============================================================
# File: scripts/daily_dashboard.py
# Purpose: Generate a daily dashboard from XGBoost runner results
# ============================================================

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date
from colorama import Fore, Style, init
import os, shutil

from core.paths import RESULTS_DIR
from core.log_config import init_global_logger

logger = init_global_logger()
init(autoreset=True)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def log_event(message: str, header: bool = False):
    """Append a timestamped event to pipeline.log."""
    with open("pipeline.log", "a") as log_file:
        if header:
            log_file.write(f"\n=== {date.today().isoformat()} Dashboard Run ===\n")
        log_file.write(f"[{datetime.now().isoformat()}] {message}\n")

def load_daily_results() -> pd.DataFrame:
    """Load today's bankroll/picks CSV."""
    today = datetime.today().strftime("%Y%m%d")
    dashboard_dir = RESULTS_DIR / "dashboard"
    csv_file = dashboard_dir / f"bankroll_{today}.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"No daily CSV found for today: {csv_file}")
    df = pd.read_csv(csv_file)
    return df

def display_actionable_picks(df: pd.DataFrame):
    """Print actionable picks in a colored table format."""
    actionable_df = df[df['is_actionable'] == True]
    if actionable_df.empty:
        print(Fore.YELLOW + "⚠️ No actionable picks today.")
        return

    print(Fore.CYAN + "=== Today's Actionable Picks ===")
    for _, row in actionable_df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        winner = row['winner']
        winner_conf = row['winner_confidence_pct']
        ou = row['ou_prediction']
        ou_line = row['ou_line']
        ou_conf = row['ou_confidence_pct']
        bet_home = row['bet_size_home']
        bet_away = row['bet_size_away']

        winner_color = Fore.GREEN if winner == home_team else Fore.RED
        print(f"{home_team} vs {away_team} | Winner: {winner_color}{winner} ({winner_conf:.1f}%)"
              f"{Style.RESET_ALL} | OU: {ou} {ou_line} ({ou_conf:.1f}%)"
              f" | Bet H:{bet_home:.2f} A:{bet_away:.2f}")

def plot_daily_bankroll(df: pd.DataFrame, season: str, timestamp: str, data_dir: Path):
    """Plot bankroll trajectory and save the figure."""
    history = df['bankroll_after'].tolist()
    chart_file = data_dir / f"bankroll_{season.replace('-', '')}_{timestamp}.png"

    plt.figure(figsize=(10, 5))
    plt.plot(history, marker='o', linestyle='-', color='blue')
    plt.title("Daily Bankroll Trajectory")
    plt.xlabel("Bets (chronological order)")
    plt.ylabel("Bankroll ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(chart_file)
    plt.close()
    logger.info(f"📈 Bankroll chart saved to {chart_file}")
    print(Fore.CYAN + f"Bankroll chart saved to {chart_file}")
    return chart_file

def save_dashboard_summary(df: pd.DataFrame, season: str, timestamp: str, data_dir: Path):
    """Save a summary of today's dashboard run into data and archives."""
    today_str = datetime.today().strftime("%Y-%m-%d")

    summary = {
        "run_date": today_str,
        "season": season,
        "bets_count": len(df),
        "bankroll_start": df['bankroll_before'].iloc[0],
        "bankroll_end": df['bankroll_after'].iloc[-1],
        "actionable_count": int(df['is_actionable'].sum())
    }
    summary_df = pd.DataFrame([summary])

    # Save daily summary
    summary_file = data_dir / f"dashboard_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"📑 Dashboard summary saved to {summary_file}")

    # Append to master run history
    run_history_file = Path("data") / "run_history.csv"
    if run_history_file.exists():
        history_df = pd.read_csv(run_history_file)
        history_df = pd.concat([history_df, summary_df], ignore_index=True)
    else:
        history_df = summary_df
    history_df.to_csv(run_history_file, index=False)
    logger.info(f"📜 Master run history updated at {run_history_file}")

    # Archive monthly
    month_str = datetime.today().strftime("%Y-%m")
    archive_dir = Path("archives") / season / month_str / "dashboard"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for file in [summary_file, run_history_file]:
        shutil.copy(file, archive_dir)
        logger.info(f"📦 Archived {os.path.basename(file)} to {archive_dir}")

# ---------------------------------------------------------
# Runner
# ---------------------------------------------------------

def run_daily_dashboard(season: str = "2025-26"):
    log_event("Dashboard run started", header=True)
    try:
        df = load_daily_results()
        display_actionable_picks(df)

        # Prepare directories
        data_dir = Path("data") / season / "dashboard"
        data_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.today().strftime("%Y-%m-%d")

        # Plot bankroll
        chart_file = plot_daily_bankroll(df, season, timestamp, data_dir)

        # Save summary + archive
        save_dashboard_summary(df, season, timestamp, data_dir)

        print(Fore.GREEN + "🏁 Dashboard run completed successfully.")
        log_event("Dashboard run completed successfully")
    except FileNotFoundError as e:
        logger.warning(str(e))
        print(Fore.YELLOW + str(e))
        log_event(f"Dashboard run failed: {e}")
    except Exception as e:
        logger.error(f"Daily dashboard failed: {e}")
        print(Fore.RED + f"❌ Daily dashboard failed: {e}")
        log_event(f"Dashboard run failed: {e}")

if __name__ == "__main__":
    run_daily_dashboard()