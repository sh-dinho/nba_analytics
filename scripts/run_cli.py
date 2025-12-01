# scripts/run_cli.py
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from app.predict_pipeline import generate_predictions
import argparse

def main():
    parser = argparse.ArgumentParser(description="NBA Analytics CLI")
    parser.add_argument("--threshold", type=float, default=0.6, help="Strong pick probability threshold")
    parser.add_argument("--strategy", choices=["kelly", "flat"], default="kelly", help="Betting strategy")
    parser.add_argument("--max_fraction", type=float, default=0.05, help="Max Kelly fraction per bet")
    args = parser.parse_args()

    # Run predictions with CLI mode
    generate_predictions(
        threshold=args.threshold,
        strategy=args.strategy,
        max_fraction=args.max_fraction
    )

if __name__ == "__main__":
    main()