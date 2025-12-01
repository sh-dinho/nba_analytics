# File: scripts/setup_all.py
import os
import subprocess

os.makedirs("results", exist_ok=True)

def run_step(cmd, desc):
    print(f"\n▶️ {desc}...")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {desc} completed")
    except subprocess.CalledProcessError as e:
        print(f"❌ {desc} failed: {e}")

def main():
    # 1. Generate player trends
    run_step(["python", "scripts/player_trends.py"], "Generate player trends")

    # 2. Build weekly summary
    run_step(["python", "scripts/build_weekly_summary.py"], "Build weekly summary")

    # 3. Train model
    run_step([
        "python", "scripts/train_model.py",
        "--seasons", "2021-22", "2022-23", "2023-24", "2024-25",
        "--train_ou"
    ], "Train model")

    # 4. Generate predictions
    run_step(["python", "app/predict_pipeline.py"], "Generate predictions")

if __name__ == "__main__":
    main()