# core/utils.py
import os
import json
from datetime import datetime

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_columns(df, required_cols, name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
