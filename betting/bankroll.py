# File: betting/bankroll.py
# Implements deterministic bankroll tracking for backtesting and daily simulation.

import pandas as pd
import numpy as np
from betting.utils import calculate_kelly_fraction 

def run_backtest(df: pd.DataFrame, initial: float = 1000.0, max_fraction: float = 0.05,
                 strategy: str = "kelly") -> tuple[pd.DataFrame, dict]:
    """
    Runs a deterministic bankroll backtest/simulation.
    
    The input DataFrame MUST have columns: 'decimal_odds', 'pred_home_win_prob', 'Date'.
    If 'won' is missing (a prediction run), a placeholder is used.
    
    Returns:
      A DataFrame with the bankroll trajectory and a dictionary of metrics.
    """
    
    df_copy = df.copy()
    
    # CRITICAL FIX: Add 'Date' and check 'won'. If 'won' is missing, it's a prediction run.
    if 'Date' not in df_copy.columns:
        df_copy['Date'] = pd.to_datetime('today').strftime('%Y-%m-%d')
        
    if 'won' not in df_copy.columns:
        # Prediction Run Mode: Use a placeholder for 'won' as the result is unknown
        df_copy['won'] = -1 
        print("⚠️ Bankroll is running in PREDICTION MODE. 'won' column is a placeholder; results are not actual.")

    bankroll = initial
    trajectory_data = []

    for i, row in df_copy.iterrows():
        p = row["pred_home_win_prob"]
        o = row["decimal_odds"]
        b = o - 1
        
        if strategy == "kelly":
            fraction = calculate_kelly_fraction(p, o, max_fraction=max_fraction)
        elif strategy == "fixed":
            fraction = max_fraction
        else:
            fraction = 0.0
            
        stake = bankroll * fraction
        
        # Use actual outcome from 'won' column (1 or 0). If -1 (prediction run), assume loss for conservative staking.
        actual_win = row['won'] if row['won'] != -1 else 0
        
        # Calculate PnL (profit or loss)
        pnl = (stake * b) if actual_win == 1 else -stake
        
        bankroll += pnl
        bankroll = max(bankroll, 0.0)

        trajectory_data.append({
            'Date': row['Date'],
            'index': i,
            'bankroll': bankroll,
            'stake': stake,
            'fraction': fraction,
            'pnl': pnl,
            'won': actual_win,
            'pred_prob': p,
            'odds': o,
        })

    trajectory_df = pd.DataFrame(trajectory_data)
    
    wins = trajectory_df['won'].sum()
    total_bets = len(trajectory_df)

    metrics = {
        "final_bankroll": bankroll,
        "roi": (bankroll - initial) / initial if initial > 0 else 0,
        "total_bets": total_bets,
        "wins": wins,
        "losses": total_bets - wins,
        "win_rate": wins / total_bets if total_bets > 0 else 0,
    }
    
    return trajectory_df, metrics