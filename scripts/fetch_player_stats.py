import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def main(use_synthetic: bool = False):
    """
    Fetch player stats. If use_synthetic=True, generate synthetic data
    for CI/CD or testing environments.
    """
    os.makedirs("data", exist_ok=True)
    out_file = "data/player_stats.csv"

    if use_synthetic:
        logger.info("⚙️ Using synthetic player stats for CI/testing...")
        df = pd.DataFrame({
            "PLAYER_NAME": ["Synthetic Player A", "Synthetic Player B"],
            "TEAM_ABBREVIATION": ["SYN", "SYN"],
            "AGE": [25, 28],
            "POSITION": ["G", "F"],
            "GAMES_PLAYED": [10, 12],
            "PTS": [15, 20],
            "AST": [5, 7],
            "REB": [4, 6],
        })
        df.to_csv(out_file, index=False)
        logger.info(f"✅ Synthetic player stats saved to {out_file}")
        return

    # --- Real scraping logic (example placeholder) ---
    try:
        logger.info("Fetching real player stats...")
        # Replace with actual scraping or API call
        df = pd.DataFrame({
            "PLAYER_NAME": ["LeBron James", "Stephen Curry"],
            "TEAM_ABBREVIATION": ["LAL", "GSW"],
            "AGE": [40, 37],
            "POSITION": ["F", "G"],
            "GAMES_PLAYED": [20, 18],
            "PTS": [25.3, 29.1],
            "AST": [7.2, 6.5],
            "REB": [8.1, 5.2],
        })
        df.to_csv(out_file, index=False)
        logger.info(f"✅ Real player stats saved to {out_file}")
    except Exception as e:
        logger.error(f"❌ Failed to fetch real stats: {e}")
        logger.info("Falling back to synthetic data...")
        main(use_synthetic=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch player stats")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Generate synthetic stats instead of scraping")
    args = parser.parse_args()
    main(use_synthetic=args.use_synthetic)