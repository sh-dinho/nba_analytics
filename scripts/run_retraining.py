# scripts/run_retraining.py
import logging
from models.train_models import train_models_cached

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting scheduled model retraining pipeline...")
    
    # This call executes the full training, evaluation, and artifact saving process.
    train_models_cached() 
    
    logging.info("Model retraining complete. New models and artifacts saved.")

if __name__ == "__main__":
    main()