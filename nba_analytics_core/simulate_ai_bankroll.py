# nba_analytics_core/simulate_ai_bankroll.py
import logging
import pandas as pd
from typing import Literal, Dict
from config import MAX_KELLY_FRACTION

def simulate_ai_strategy(initial_bankroll: float, strategy: Literal["flat", "kelly"] = "flat") -> pd.DataFrame:
    logging.info(f"Simulating strategy={strategy} with initial_bankroll={initial_bankroll}")
    # Placeholder: using fabricated odds and probabilities
    data = pd.DataFrame({
        "game_id": ["2025-001", "2025-002", "2025-003"],
        "team": ["LAL", "BOS", "NYK"],
        "decimal_odds": [1.9, 2.1, 1.8],
        "prob": [0.60, 0.58, 0.64],
        "realized": [True, False, True],
    })

    bankroll = initial_bankroll
    stakes = []
    pnl = []
    bankroll_series = []

    for _, row in data.iterrows():
        if strategy == "flat":
            stake = min(0.02 * bankroll, bankroll)  # 2% flat stake
        else:  # kelly
            b = row["decimal_odds"] - 1
            kelly_f = (row["prob"] * (b + 1) - 1) / b if b > 0 else 0
            stake = max(0.0, min(kelly_f, MAX_KELLY_FRACTION)) * bankroll

        outcome = (row["decimal_odds"] - 1) * stake if row["realized"] else -stake
        bankroll += outcome

        stakes.append(stake)
        pnl.append(outcome)
        bankroll_series.append(bankroll)

    data["stake"] = stakes
    data["pnl"] = pnl
    data["bankroll"] = bankroll_series

    # Attach KPIs
    roi = (bankroll - initial_bankroll) / initial_bankroll if initial_bankroll > 0 else 0.0
    win_rate = data["realized"].mean() if not data.empty else 0.0
    max_drawdown = float((pd.Series(bankroll_series).cummax() - pd.Series(bankroll_series)).max() or 0.0)
    kpis: Dict[str, float] = {"roi": float(roi), "win_rate": float(win_rate), "max_drawdown": float(max_drawdown)}
    setattr(data, "kpis", kpis)
    logging.info(f"✔ Simulation complete: ROI={roi:.3f}, WinRate={win_rate:.3f}, MaxDD={max_drawdown:.2f}")
    return data