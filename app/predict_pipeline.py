from scripts.simulate_bankroll import simulate_bankroll

def generate_predictions(threshold=0.6, strategy="kelly", max_fraction=0.05):
    odds_data = fetch_odds()
    games = fetch_historical_games("2024-25")
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

    # --- Print best picks ---
    best_picks = df.loc[df["strong_pick"] == 1].sort_values("home_ev", ascending=False)
    if not best_picks.empty:
        print("\n=== BEST PICKS ===")
        for _, row in best_picks.iterrows():
            print(f"{row['home_team']} vs {row['away_team']} | Prob={row['pred_home_win_prob']:.2f} | EV={row['home_ev']:.3f}")
    else:
        print("\nNo strong picks found today.")

    # --- Simulate bankroll and print metrics ---
    bets = best_picks.rename(columns={
        "home_team": "team",
        "home_decimal_odds": "decimal_odds",
        "pred_home_win_prob": "prob",
        "home_ev": "ev"
    })[["team", "decimal_odds", "prob", "ev"]]

    if not bets.empty:
        _, metrics = simulate_bankroll(bets, strategy=strategy, max_fraction=max_fraction)
        print("\n=== BANKROLL METRICS ===")
        print(f"Final Bankroll: ${metrics['final_bankroll']:.2f}")
        print(f"ROI: {metrics['roi']*100:.2f}%")
        print(f"Win Rate: {metrics['win_rate']*100:.1f}% ({metrics['wins']}W/{metrics['losses']}L)")

    return df