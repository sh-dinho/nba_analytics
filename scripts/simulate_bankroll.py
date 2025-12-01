# app/predict_pipeline.py
import joblib
import pandas as pd
from nba_analytics_core.odds import fetch_odds
from scripts.train_model import build_team_stats, build_matchup_features, fetch_historical_games

# Always load the latest model at startup
MODEL_PATH = "models/classification_model.pkl"
model = joblib.load(MODEL_PATH)

def generate_predictions(threshold=0.6):
    odds_data = fetch_odds()
    games = fetch_historical_games("2024-25")  # or use multiple seasons
    team_stats = build_team_stats(games)

    rows = []
    for game in odds_data:
        home = game["home_team"]
        away = game["away_team"]

        features = build_matchup_features(home, away, team_stats)
        pred_home_win_prob = model.predict_proba([features])[0][1]

        home_odds = game["bookmakers"][0]["markets"][0]["outcomes"][0]["price"]
        ev = (pred_home_win_prob * (home_odds - 1)) - (1 - pred_home_win_prob)

        rows.append({
            "home_team": home,
            "away_team": away,
            "pred_home_win_prob": pred_home_win_prob,
            "home_decimal_odds": home_odds,
            "home_ev": ev,
            "strong_pick": 1 if pred_home_win_prob >= threshold and ev > 0 else 0,
        })
    df = pd.DataFrame(rows)

    # --- Print best picks in command line ---
    best_picks = df.loc[df["strong_pick"] == 1].sort_values("home_ev", ascending=False)
    if not best_picks.empty:
        print("\n=== BEST PICKS ===")
        for _, row in best_picks.iterrows():
            print(f"{row['home_team']} vs {row['away_team']} | Prob={row['pred_home_win_prob']:.2f} | EV={row['home_ev']:.3f}")
    else:
        print("\nNo strong picks found today.")

    return df