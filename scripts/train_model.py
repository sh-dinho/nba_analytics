import argparse
import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from nba_analytics_core.data import fetch_historical_games, build_team_stats, build_matchup_features
from nba_analytics_core.odds import fetch_odds
from scripts.simulate_bankroll import simulate_bankroll

MODEL_PATH = "models/classification_model.pkl"

def train_model(seasons):
    X, y = [], []
    for season in seasons:
        games = fetch_historical_games(season)
        team_stats = build_team_stats(games)
        for g in games:
            if not g.get("away_team"):
                continue
            feats = build_matchup_features(g["home_team"], g["away_team"], team_stats)
            X.append(feats)
            y.append(1 if g["home_win"] else 0)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model trained and saved to {MODEL_PATH}")
    return model

def backtest_model(seasons, strategy="kelly", max_fraction=0.05, export_path="results/backtest.csv"):
    model = joblib.load(MODEL_PATH)
    results = []

    for season in seasons:
        games = fetch_historical_games(season)
        team_stats = build_team_stats(games)
        bets = []

        for g in games:
            if not g.get("away_team"):
                continue
            feats = build_matchup_features(g["home_team"], g["away_team"], team_stats)
            prob = float(model.predict_proba([feats])[0][1])

            odds = fetch_odds(home_team=g["home_team"], away_team=g["away_team"])
            if not odds:
                continue
            home_odds = odds["home_odds"]
            ev = (prob * (home_odds - 1)) - (1 - prob)

            bets.append({"team": g["home_team"], "decimal_odds": home_odds, "prob": prob, "ev": ev})

        if not bets:
            continue

        _, metrics = simulate_bankroll(pd.DataFrame(bets), strategy=strategy, max_fraction=max_fraction)
        results.append({
            "season": season,
            "roi": metrics["roi"],
            "win_rate": metrics["win_rate"],
            "final_bankroll": metrics["final_bankroll"]
        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    df.to_csv(export_path, index=False)
    print(f"📂 Backtest results exported to {export_path}")

def main():
    parser = argparse.ArgumentParser(description="Train and backtest NBA model")
    parser.add_argument("--seasons", nargs="+", default=["2021-22","2022-23","2023-24"], help="Seasons to train/backtest on")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting after training")
    parser.add_argument("--export", type=str, default="results/backtest.csv", help="Path to export backtest results")
    parser.add_argument("--strategy", choices=["kelly", "flat"], default="kelly")
    parser.add_argument("--max_fraction", type=float, default=0.05)
    args = parser.parse_args()

    train_model(args.seasons)
    if args.backtest:
        backtest_model(args.seasons, strategy=args.strategy, max_fraction=args.max_fraction, export_path=args.export)

if __name__ == "__main__":
    main()