import argparse
from pathlib import Path
from src.model_training.train_combined import train_model
from prediction_engine.daily_runner_mlflow import daily_runner_mlflow
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

MODEL_PATH = cfg["model_path"]
MODEL_TYPE = cfg["model_type"]
CACHE_PATH = Path("data/cache/features_full.parquet")

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--date", help="YYYY-MM-DD for daily run")
args = parser.parse_args()

if args.train:
    train_model(CACHE_PATH, out_dir=Path("models"), model_type=MODEL_TYPE)
else:
    date = args.date or "today"
    df = daily_runner_mlflow(MODEL_PATH, date)
    print(df.head())
