# ============================================================
# File: scripts/daily_dashboard.py
# Purpose: Generate a daily dashboard from XGBoost runner results
# ============================================================

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from colorama import Fore, Style, init

from core.paths import RESULTS_DIR
from core.log_config import init_global_logger

logger = init_global_logger()
init(autoreset=True)

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

def plot_daily_bankroll(df: pd.DataFrame):
    """Plot bankroll trajectory and save the figure."""
    history = df['bankroll_after'].tolist()
    today = datetime.today().strftime("%Y%m%d")
    dashboard_dir = RESULTS_DIR / "dashboard"
    chart_file = dashboard_dir / f"bankroll_{today}.png"

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

def run_daily_dashboard():
    try:
        df = load_daily_results()
        display_actionable_picks(df)
        plot_daily_bankroll(df)
    except FileNotFoundError as e:
        logger.warning(str(e))
        print(Fore.YELLOW + str(e))
    except Exception as e:
        logger.error(f"Daily dashboard failed: {e}")
        print(Fore.RED + f"❌ Daily dashboard failed: {e}")

if __name__ == "__main__":
    run_daily_dashboard()
