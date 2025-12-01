# scripts/simulate_bankroll.py
import pandas as pd

def kelly_fraction(p: float, decimal_odds: float) -> float:
    # Convert decimal odds to net odds b (profit per 1 unit)
    b = decimal_odds - 1.0
    # Kelly: f* = (bp - (1-p)) / b
    return max(0.0, (b * p - (1 - p)) / b) if b > 0 else 0.0

def simulate_bankroll(df_bets: pd.DataFrame, initial_bankroll: float = 1000.0, strategy: str = "kelly", max_fraction: float = 0.05) -> pd.DataFrame:
    """
    df_bets columns: game_id, team, decimal_odds, prob, result (0/1 or None)
    """
    if df_bets.empty:
        return pd.DataFrame()

    bankroll = initial_bankroll
    records = []
    for _, b in df_bets.iterrows():
        if strategy == "kelly":
            f = kelly_fraction(b["prob"], b["decimal_odds"])
            stake = min(max_fraction, f) * bankroll
        elif strategy == "flat":
            stake = 0.01 * bankroll  # 1% flat
        else:
            stake = 0.01 * bankroll

        # If result unknown (game not finished), assume unrealized; keep bankroll static
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

    return pd.DataFrame(records)