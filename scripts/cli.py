import argparse
from app.prediction_pipeline import PredictionPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Analytics CLI")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--strategy", choices=["kelly","flat"], default="kelly")
    parser.add_argument("--max_fraction", type=float, default=0.05)
    args = parser.parse_args()

    pipeline = PredictionPipeline(
        threshold=args.threshold,
        strategy=args.strategy,
        max_fraction=args.max_fraction
    )
    df, metrics = pipeline.run()
    if metrics:
        print("✅ CLI run completed")
