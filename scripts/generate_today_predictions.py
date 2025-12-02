import pandas as pd
import os

def generate_today_predictions(outdir="results", threshold=0.6):
    print(f"📈 Generating today's predictions with threshold {threshold}")
    # Example predictions dataframe
    df = pd.DataFrame({
        "game_id": [1, 2],
        "home_team": ["LAL", "BOS"],
        "away_team": ["NYK", "MIA"],
        "pred_home_win_prob": [0.65, 0.70],
        "decimal_odds": [1.90, 1.85],
        "ev": [0.12, 0.15]
    })
    os.makedirs(outdir, exist_ok=True)
    preds_file = os.path.join(outdir, "predictions.csv")
    df.to_csv(preds_file, index=False)
    print(f"✅ Predictions saved to {preds_file}")
    return df
