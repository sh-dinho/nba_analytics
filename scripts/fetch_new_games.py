import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def main(use_synthetic: bool = False):
    """
    Fetch today's games. If use_synthetic=True, generate synthetic games for CI/testing.
    Saves to data/new_games.csv with the same feature columns as training.
    """
    os.makedirs("data", exist_ok=True)
    out_file = "data/new_games.csv"

    if use_synthetic:
        logger.info("⚙️ Using synthetic new_games.csv for CI/testing...")
        df = pd.DataFrame({
            "TEAM_HOME": ["SYN_A", "SYN_B"],
            "TEAM_AWAY": ["SYN_C", "SYN_D"],
            "AGE": [25, 28],
            "PTS": [15, 20],
            "AST": [5, 7],
            "REB": [4, 6],
            "GAMES_PLAYED": [10, 12],
            "decimal_odds": [1.8, 2.1],
        })
        df.to_csv(out_file, index=False)
        logger.info(f"✅ Synthetic new_games.csv saved to {out_file}")
        return

    # --- Real scraping logic placeholder ---
    try:
        logger.info("Fetching real games...")
        # Replace with actual scraping/API call
        df = pd.DataFrame({
            "TEAM_HOME": ["LAL", "GSW"],
            "TEAM_AWAY": ["BOS", "MIA"],
            "AGE": [30, 27],
            "PTS": [25, 29],
            "AST": [7, 6],
            "REB": [8, 5],
            "GAMES_PLAYED": [20, 18],
            "decimal_odds": [1.9, 2.2],
        })
        df.to_csv(out_file, index=False)
        logger.info(f"✅ Real new_games.csv saved to {out_file}")
    except Exception as e:
        logger.error(f"❌ Failed to fetch real games: {e}")
        logger.info("Falling back to synthetic games...")
        main(use_synthetic=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch today's games")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Generate synthetic games instead of scraping")
    args = parser.parse_args()
    main(use_synthetic=args.use_synthetic)