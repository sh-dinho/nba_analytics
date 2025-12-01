# scripts/simulate_bankroll.py (Updated with KPI Summary)
import pandas as pd

def kelly_fraction(p: float, decimal_odds: float) -> float:
    # Convert decimal odds to net odds b (profit per 1 unit)
    b = decimal_odds - 1.0
    # Kelly: f* = (bp - (1-p)) / b
    return max(0.0, (b * p - (1 - p)) / b) if b > 0 else 0.0

def calculate_kpis(records: pd.DataFrame, initial_bankroll: float) -> dict:
    """Calculates summary KPIs from the simulation results."""
    if records.empty:
        return { "roi": 0.0, "win_rate": 0.0, "max_drawdown": 0.0, "final_bankroll": initial_bankroll, "total_bets": 0 }

    final_bankroll = records['bankroll'].iloc[-1]
    total_bets = records['realized'].sum()

    if total_bets == 0:
        return { "roi": 0.0, "win_rate": 0.0, "max_drawdown": 0.0, "final_bankroll": final_bankroll, "total_bets": 0 }

    # ROI
    roi = (final_bankroll - initial_bankroll) / initial_bankroll

    # Win Rate (Realized bets only)
    total_pnl = records[records['realized'] == True]['pnl']
    wins = (total_pnl > 0).sum()
    win_rate = wins / total_bets

    # Max Drawdown
    bankroll_history = records['bankroll']
    cumulative_max = bankroll_history.cummax()
    drawdown = cumulative_max - bankroll_history
    max_drawdown = (drawdown / cumulative_max.replace(0, 1)).max()

    return {
        "roi": roi,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "final_bankroll": final_bankroll,
        "total_bets": int(total_bets)
    }


def simulate_bankroll(df_bets: pd.DataFrame, initial_bankroll: float = 1000.0, strategy: str = "kelly", max_fraction: float = 0.05) -> pd.DataFrame:
    """
    Simulates the bankroll progression and returns the history DataFrame along 
    with a dictionary of key performance indicators (KPIs) attached as attributes.
    """
    if df_bets.empty:
        df_history = pd.DataFrame()
        df_history.kpis = calculate_kpis(df_history, initial_bankroll) 
        return df_history

    bankroll = initial_bankroll
    records = []
    for _, b in df_bets.iterrows():
        # ... (Staking Logic remains the same) ...
        if strategy == "kelly":
            f = kelly_fraction(b["prob"], b["decimal_odds"])
            stake = min(max_fraction, f) * bankroll
        elif strategy == "flat":
            stake = 0.01 * bankroll 
        else:
            stake = 0.01 * bankroll

        # ... (P&L and Bankroll Update Logic remains the same) ...
        if pd.isna(b["result"]):
            pnl = 0.0
            realized = False
        else:
            realized = True
            if int(b["result"]) == 1:
                pnl = stake * (b["decimal_odds"] - 1.0)
            else:
                pnl = -stake
            bankroll += pnl

        records.append({
            "game_id": b["game_id"],
            "team": b["team"],
            "decimal_odds": b["decimal_odds"],
            "prob": b["prob"],
            "stake": stake,
            "pnl": pnl,
            "realized": realized,
            "bankroll": bankroll
        })

    df_history = pd.DataFrame(records)
    
    # --- Calculate and Attach KPIs ---
    df_history.kpis = calculate_kpis(df_history, initial_bankroll)
    
    return df_history