# ============================================================
# File: src/features/feature_store.py
# ============================================================

from pathlib import Path
from datetime import datetime
import pandas as pd
from loguru import logger


class FeatureStore:
    def __init__(self):
        self.base_dir = Path("data/features")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, df: pd.DataFrame, kind: str = "training") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"features_{timestamp}_{kind}.parquet"
        path = self.base_dir / filename
        df.to_parquet(path, index=False)
        logger.info(f"Saved feature snapshot → {path} ({len(df)} rows)")
        return str(path)

    def load_latest_snapshot(self):
        files = sorted(self.base_dir.glob("features_*.parquet"))
        if not files:
            raise FileNotFoundError("No feature snapshots found in data/features/")
        latest = files[-1]
        df = pd.read_parquet(latest)
        meta = {
            "path": str(latest),
            "rows": len(df),
            "columns": list(df.columns),
        }
        logger.info(f"Loaded feature snapshot: {latest} ({len(df)} rows)")
        return df, meta
