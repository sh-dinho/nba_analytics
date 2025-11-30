import logging
from datetime import datetime
import sqlite3
import yaml

logging.basicConfig(level=logging.INFO)
CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]

def suggest_stake(team_win_percentage, current_bankroll, confidence_factor=None, min_stake=None):
    cf = confidence_factor if confidence_factor is not None else CONFIG["app"]["confidence_factor"]
    ms = min_stake if min_stake is not None else CONFIG["app"]["min_stake"]
    if not (0 <= team_win_percentage <= 1):
        raise ValueError("team_win_percentage must be between 0 and 1")
    if current_bankroll <= 0:
        return 0.0
    stake = current_bankroll * cf * team_win_percentage
    return max(stake, ms)

def suggest_stake_kelly(team_win_percentage, odds, current_bankroll, min_stake=None):
    ms = min_stake if min_stake is not None else CONFIG["app"]["min_stake"]
    if not (0 <= team_win_percentage <= 1):
        raise ValueError("team_win_percentage must be between 0 and 1")
    if current_bankroll <= 0:
        return 0.0
    b = odds - 1
    p = team_win_percentage
    q = 1 - p
    kelly_fraction = (b * p - q) / b if b > 0 else 0
    if kelly_fraction <= 0:
        return 0.0
    stake = current_bankroll * kelly_fraction
    return max(stake, ms)

def simulate_bet(outcome, stake, odds):
    if outcome.lower() == "win":
        return stake * (odds - 1)  # Profit only
    elif outcome.lower() == "loss":
        return -stake
    else:
        raise ValueError("Outcome must be 'win' or 'loss'")

def update_bankroll_and_roi(stake, outcome, odds, current_bankroll):
    profit_or_loss = simulate_bet(outcome, stake, odds)
    new_bankroll = current_bankroll + profit_or_loss
    roi_per_bet = profit_or_loss / stake if stake > 0 else 0
    bankroll_growth = (new_bankroll - current_bankroll) / current_bankroll if current_bankroll > 0 else 0
    return new_bankroll, roi_per_bet, bankroll_growth

def track_bet(bet_team, stake, odds, outcome, new_bankroll, roi):
    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            cur.execute("""
                INSERT INTO bet_tracking (bet_timestamp, bet_team, stake, odds, outcome, bankroll, roi)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                bet_team.strip() if bet_team else "Unknown",
                float(stake),
                float(odds),
                outcome.lower(),
                float(new_bankroll),
                float(roi)
            ))
            con.commit()
        logging.info(
            f"✔ Bet tracked: Team={bet_team}, Stake={stake:.2f}, Odds={odds}, "
            f"Outcome={outcome}, Bankroll={new_bankroll:.2f}, ROI={roi:.2%}"
        )
    except sqlite3.Error as e:
        logging.error(f"❌ Database error while tracking bet: {e}")
    except Exception as e:
        logging.error(f"❌ Unexpected error while tracking bet: {e}")