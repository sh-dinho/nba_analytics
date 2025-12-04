# ============================================================
# File: modeling/simulate_games.py
# Purpose: Run Monte Carlo simulations of upcoming games using ensemble model
# ============================================================

import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from core.config import (
    ENSEMBLE_MODEL_FILE,
    NEW_GAMES_FEATURES_FILE,
    BASE_RESULTS_DIR,
    SUMMARY_FILE as PIPELINE_SUMMARY_FILE
)
from core.log_config import init_global_logger
from core.exceptions import PipelineError
from notifications import send_message, send_photo

logger = init_global_logger()

def load_model():
    try:
        return joblib.load(ENSEMBLE_MODEL_FILE)
    except Exception as e:
        raise PipelineError(f"Failed to load ensemble model: {e}")

def load_games():
    try:
        return pd.read_csv(NEW_GAMES_FEATURES_FILE)
    except Exception as e:
        raise PipelineError(f"Failed to load new games features: {e}")

def simulate_game(model, game_features, n_simulations=1000):
    """Run Monte Carlo simulation for a single game with noise injection."""
    X = game_features.values.reshape(1, -1)
    probs = []
    for _ in range(n_simulations):
        noisy_X = X + np.random.normal(0, 0.05, X.shape)
        prob = model.predict_proba(noisy_X)[0, 1]
        probs.append(prob)
    return probs

def run_simulations(model, games, n_simulations=1000):
    results = []
    sim_dir = BASE_RESULTS_DIR / "simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in games.iterrows():
        game_id = row.get("GAME_ID", idx)
        features = row.drop(labels=["GAME_ID"], errors="ignore")
        probs = simulate_game(model, features, n_simulations)
        mean_prob = np.mean(probs)
        std_prob = np.std(probs)
        results.append({
            "game_id": game_id,
            "mean_win_prob": mean_prob,
            "std_dev": std_prob
        })

        # Save histogram plot
        plt.hist(probs, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        plt.title(f"Simulation for Game {game_id}")
        plt.xlabel("Win Probability")
        plt.ylabel("Frequency")
        plt.grid(True)
        plot_path = sim_dir / f"game_{game_id}_simulation.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"📊 Saved simulation plot → {plot_path}")

    return pd.DataFrame(results)

def main(n_simulations, season="aggregate", notes="simulation run",
         export_json=False, overwrite=False, notify=False):
    logger.info("🚀 Starting Monte Carlo simulations...")
    model = load_model()
    games = load_games()
    results = run_simulations(model, games, n_simulations)

    sim_dir = BASE_RESULTS_DIR / "simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)

    output_file = sim_dir / "simulation_results.csv"
    results.to_csv(output_file, index=False)
    logger.info(f"✅ Simulation results saved → {output_file}")

    if export_json:
        results.to_json(sim_dir / "simulation_results.json", orient="records", indent=2)
        logger.info("📑 Simulation results also exported to JSON")

    # Append or overwrite centralized pipeline_summary.csv
    run_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    results["timestamp"] = run_time
    results["season"] = season
    results["target"] = "simulation"
    results["model_type"] = "ensemble"
    results["notes"] = notes

    if overwrite or not PIPELINE_SUMMARY_FILE.exists():
        results.to_csv(PIPELINE_SUMMARY_FILE, index=False)
        logger.info(f"📑 Centralized summary OVERWRITTEN at {PIPELINE_SUMMARY_FILE}")
    else:
        results.to_csv(PIPELINE_SUMMARY_FILE, mode="a", header=False, index=False)
        logger.info(f"📑 Simulation results appended to {PIPELINE_SUMMARY_FILE}")

    if notify:
        send_message(f"🤖 Simulation run complete ({season}) — results saved.")
        try:
            send_photo(str(output_file), caption="📊 Simulation Results CSV")
        except Exception:
            logger.warning("⚠️ Could not send CSV file to Telegram")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo simulation of upcoming games")
    parser.add_argument("--n_simulations", type=int, default=1000,
                        help="Number of Monte Carlo simulations per game")
    parser.add_argument("--season", type=str, default="aggregate", help="Season tag for simulation entries")
    parser.add_argument("--notes", type=str, default="simulation run", help="Optional notes to annotate simulation entries")
    parser.add_argument("--export-json", action="store_true", help="Also export simulation results as JSON")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite centralized pipeline_summary.csv instead of appending")
    parser.add_argument("--notify", action="store_true", help="Send simulation summary to Telegram")
    args = parser.parse_args()

    main(args.n_simulations, season=args.season, notes=args.notes,
         export_json=args.export_json, overwrite=args.overwrite, notify=args.notify)
