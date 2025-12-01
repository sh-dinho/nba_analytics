import pandas as pd

def simulate_bankroll(df, strategy="kelly", max_fraction=0.05, initial_bankroll=1000):
    """
    Simulate bankroll growth given bets.
    df requires: ['decimal_odds', 'prob', 'ev'].
    Returns (bankroll_history, metrics).
    """
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    wins, losses = 0, 0

    for _, bet in df.iterrows():
        prob = bet["prob"]
        odds = bet["decimal_odds"]

        if strategy == "kelly":
            k = ((prob * (odds - 1)) - (1 - prob)) / (odds - 1)
            stake = bankroll * max(0, min(k, max_fraction))
        else:
            stake = bankroll * 0.02  # flat 2%

        # naive: bet outcome based on probability threshold 0.5
        outcome_win = prob >= 0.5
        if outcome_win:
            bankroll += stake * (odds - 1)
            wins += 1
        else:
            bankroll -= stake
            losses += 1

        bankroll_history.append(bankroll)

    roi = (bankroll - initial_bankroll) / initial_bankroll
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    metrics = {
        "final_bankroll": bankroll,
        "roi": roi,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses
    }
    return bankroll_history, metrics