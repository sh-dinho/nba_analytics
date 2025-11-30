from app.predictor import predict_todays_games
from simulate_bankroll import simulate_bankroll
from odds_api import fetch_live_odds  # new function above

def simulate_ai_strategy(initial_bankroll=1000, strategy="kelly"):
    df_pred = predict_todays_games()
    df_odds = fetch_live_odds()

    # Merge predictions with odds
    df = df_pred.merge(df_odds, on=["home_team","away_team"])

    bets = []
    for _, row in df.iterrows():
        if row["pred_home_win_prob"] > 0.55:
            bets.append({
                "game_id": row["game_id"],
                "team": row["home_team"],
                "odds": row["odds_home"],
                "stake": 1.0,
                "result": 1 if row["winner"] == row["home_team"] else 0
            })

    df_bets = pd.DataFrame(bets)
    return simulate_bankroll(df_bets, initial_bankroll=initial_bankroll, strategy=strategy)