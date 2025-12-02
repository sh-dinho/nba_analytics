from app.prediction_pipeline import PredictionPipeline

if __name__ == "__main__":
    pipeline = PredictionPipeline(threshold=0.6, strategy="kelly", max_fraction=0.05)
    df, metrics = pipeline.run()

    if metrics:
        print("✅ Daily pipeline completed successfully")
        print(f"Final Bankroll: {metrics['final_bankroll_mean']:.2f}")
        print(f"ROI: {metrics['roi']*100:.2f}%")
