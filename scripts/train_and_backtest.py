# File: scripts/train_and_backtest.py
import argparse
import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from nba_analytics_core.data import fetch_historical_games, build_team_stats, build_matchup_features
from nba_analytics_core.odds import fetch_odds
from scripts.simulate_bankroll import simulate_bankroll
from nba_analytics_core.notifications import send_telegram_message

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


def backtest_model(seasons, strategy="kelly", max_fraction=0.05,
                   export_summary="results/backtest_summary.csv",
                   export_detailed="results/backtest_detailed.csv",
                   notify=False):
    model = joblib.load(MODEL_PATH)
    summary_results, detailed_results = [], []

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

            bets.append({
                "season": season,
                "game_id": g.get("game_id"),
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "decimal_odds": home_odds,
                "prob": prob,
                "ev": ev,
                "home_win": g["home_win"]
            })

        if not bets:
            continue

        df_bets = pd.DataFrame(bets)
        _, metrics = simulate_bankroll(df_bets, strategy=strategy, max_fraction=max_fraction)

        # Compute per-season metrics
        acc = accuracy_score(df_bets["home_win"], (df_bets["prob"] >= 0.5).astype(int))
        ll = log_loss(df_bets["home_win"], df_bets["prob"])
        brier = brier_score_loss(df_bets["home_win"], df_bets["prob"])

        summary_results.append({
            "season": season,
            "roi": metrics["roi"],
            "win_rate": metrics["win_rate"],
            "final_bankroll": metrics["final_bankroll"],
            "accuracy": acc,
            "log_loss": ll,
            "brier": brier
        })
        detailed_results.extend(bets)

    # Export results
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(summary_results).to_csv(export_summary, index=False)
    pd.DataFrame(detailed_results).to_csv(export_detailed, index=False)
    print(f"📂 Backtest summary exported to {export_summary}")
    print(f"📂 Detailed bets exported to {export_detailed}")

    # Telegram notification
    if notify and summary_results:
        avg_roi = pd.DataFrame(summary_results)["roi"].mean()
        avg_acc = pd.DataFrame(summary_results)["accuracy"].mean()
        msg = (
            f"📊 Backtest Summary\n"
            f"Seasons: {', '.join(seasons)}\n"
            f"Games: {len(detailed_results)}\n"
            f"Avg ROI: {avg_roi:.2f}\n"
            f"Avg Accuracy: {avg_acc:.2f}"
        )
        send_telegram_message(msg)


def main():
    parser = argparse.ArgumentParser(description="Train and backtest NBA model")
    parser.add_argument("--seasons", nargs="+", default=["2021-22","2022-23","2023-24"], help="Seasons to train/backtest on")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting after training")
    parser.add_argument("--export_summary", type=str, default="results/backtest_summary.csv", help="Path to export backtest summary")
    parser.add_argument("--export_detailed", type=str, default="results/backtest_detailed.csv", help="Path to export detailed bets")
    parser.add_argument("--strategy", choices=["kelly", "flat"], default="kelly")
    parser.add_argument("--max_fraction", type=float, default=0.05)
    parser.add_argument("--notify", action="store_true", help="Send Telegram notification after backtest")
    args = parser.parse_args()

    train_model(args.seasons)
    if args.backtest:
        backtest_model(args.seasons,
                       strategy=args.strategy,
                       max_fraction=args.max_fraction,
                       export_summary=args.export_summary,
                       export_detailed=args.export_detailed,
                       notify=args.notify)


if __name__ == "__main__":
    main()