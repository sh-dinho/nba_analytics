# File: scripts/setup_all.py
import os
import subprocess
import datetime

os.makedirs("results", exist_ok=True)

# Create a unique log file per run
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = f"setup_log_{timestamp}.txt"

def log(message):
    """Write message to both console and log file."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_step(cmd, desc):
    log(f"▶️ {desc}...")
    try:
        subprocess.run(cmd, check=True)
        log(f"✅ {desc} completed")
        return True
    except subprocess.CalledProcessError as e:
        log(f"❌ {desc} failed: {e}")
        return False
    except FileNotFoundError:
        log(f"❌ {desc} failed: script not found")
        return False

def main():
    steps = [
        (["python", "scripts/player_trends.py"], "Generate player trends"),
        (["python", "scripts/build_weekly_summary.py"], "Build weekly summary"),
        ([
            "python", "scripts/train_model.py",
            "--seasons", "2021-22", "2022-23", "2023-24", "2024-25",
            "--train_ou"
        ], "Train model"),
        (["python", "app/predict_pipeline.py"], "Generate predictions"),
    ]

    results = {}
    for cmd, desc in steps:
        success = run_step(cmd, desc)
        results[desc] = success

    log("\n📊 Summary:")
    for desc, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        log(f"- {desc}: {status}")

    log(f"\nRun complete. Full log saved to {LOG_FILE}")

if __name__ == "__main__":
    main()