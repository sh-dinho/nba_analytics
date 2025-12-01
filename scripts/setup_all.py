# File: scripts/setup_all.py
import os
import pandas as pd
import joblib

# Import your functions from other scripts
from scripts.generate_weekly_snapshots import main as generate_snapshots
from scripts.weekly_summary import main as generate_summary
from scripts.player_trends import main as generate_trends
from scripts.build_training_data import main as build_training
from scripts.train_model import main as train_model
from app.predict_pipeline import generate_today_predictions
from scripts.generate_picks import main as generate_picks

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Step 1: Generate weekly snapshots
    print("📸 Generating weekly snapshots...")
    generate_snapshots()

    # Step 2: Build weekly summary
    print("📊 Building weekly summary...")
    generate_summary()

    # Step 3: Build player trends
    print("📈 Building player trends...")
    generate_trends()

    # Step 4: Build training dataset
    print("🛠️ Building training dataset...")
    build_training()

    # Step 5: Train model
    print("🤖 Training model...")
    train_model()

    # Step 6: Predict today’s games
    print("🔮 Generating today's predictions...")
    df = generate_today_predictions()
    preds_file = "results/predictions.csv"
    df.to_csv(preds_file, index=False)
    print(f"✅ Predictions saved to {preds_file}")

    # Step 7: Generate picks
    print("🏀 Generating picks...")
    generate_picks()
    print("✅ Picks saved to results/picks.csv")

if __name__ == "__main__":
    main()