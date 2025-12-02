# File: run_pipeline.ps1
# Purpose: Run full NBA analytics pipeline locally in one go (PowerShell script)

# Ensure PYTHONPATH includes project root
$env:PYTHONPATH = "$pwd;" + $env:PYTHONPATH

Write-Host "📥 Step 1: Fetching player stats... (Creates data/player_stats.csv)"
python scripts/fetch_player_stats.py

Write-Host "🛠 Step 2: Building training features... (Creates data/training_features.csv)"
python scripts/build_features.py

Write-Host "🧠 Step 3: Training Model... (Creates models/game_predictor.pkl)"
# 🌟 CRITICAL FIX: This step was missing and caused the initial FileNotFoundError
python scripts/train_model.py

Write-Host "🚀 Step 4: Running full daily pipeline (Predict, Pick, Simulate)..."
# This CLI assumes the model file now exists and runs the rest of the steps.
python scripts/run_daily_pipeline_cli.py --threshold 0.6 --strategy kelly --max_fraction 0.05

Write-Host "✅ Pipeline complete. Check 'results/' and 'data/' directories for outputs."