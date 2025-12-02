# File: run_pipeline.ps1
# Purpose: Run full NBA analytics pipeline locally in one go

# Ensure PYTHONPATH includes project root
$env:PYTHONPATH = "$pwd;" + $env:PYTHONPATH

Write-Host "📥 Fetching player stats..."
# Step 1: Fetches stats and creates data/player_stats.csv
python scripts/fetch_player_stats.py

Write-Host "🛠 Building training features..."
# Step 2: Consumes stats, creates data/training_features.csv (Your submitted script)
python scripts/build_features.py

Write-Host "🧠 Training Model..."
# Step 3: CRITICAL MISSING STEP. Consumes features, creates models/game_predictor.pkl
python scripts/train_model.py

Write-Host "🚀 Running pipeline CLI..."
# Step 4: Runs predictions, picks, and bankroll simulation
python scripts/run_daily_pipeline_cli.py --threshold 0.6 --strategy kelly --max_fraction 0.05

Write-Host "✅ Pipeline complete. Check 'results/' and 'logs/' directories for outputs."