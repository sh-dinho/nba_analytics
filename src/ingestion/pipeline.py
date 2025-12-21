# ============================================================
# File: src/ingestion/pipeline.py
# ============================================================

from pathlib import Path
import pandas as pd
from loguru import logger

from src.ingestion.collector import NBADataCollector
from src.ingestion.transform import wide_to_long
from src.config import SNAPSHOT_PATH


class IngestionPipeline:
    def __init__(self):
        self.snapshot_path = SNAPSHOT_PATH
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.collector = NBADataCollector()

    # ---------------------------------------------------------
    # Snapshot helpers
    # ---------------------------------------------------------

    def _load_snapshot(self) -> pd.DataFrame:
        if not self.snapshot_path.exists():
            logger.info("No snapshot found. Starting fresh.")
            return pd.DataFrame()
        return pd.read_parquet(self.snapshot_path)

    def _save_snapshot(self, df: pd.DataFrame):
        if df.empty:
            logger.warning("Attempted to save empty snapshot. Skipping.")
            return
        df = df.sort_values("date")
        df.to_parquet(self.snapshot_path, index=False)
        logger.success(f"Saved snapshot with {len(df)} games → {self.snapshot_path}")

    # ---------------------------------------------------------
    # Full history ingestion
    # ---------------------------------------------------------

    def run_full_history_ingestion(self) -> pd.DataFrame:
        logger.info("🚀 Running full history ingestion...")
        df = self.collector.fetch_all_history()
        self._save_snapshot(df)
        return df

    # ---------------------------------------------------------
    # Today ingestion
    # ---------------------------------------------------------

    def run_today_ingestion(self) -> pd.DataFrame:
        logger.info("🔄 Running today's ingestion...")
        base = self._load_snapshot()

        today_df = self.collector.fetch_today()
        if today_df.empty:
            logger.info("No games today.")
            return pd.DataFrame()

        combined = pd.concat([base, today_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["game_id"], keep="last")

        self._save_snapshot(combined)
        return today_df

    # ---------------------------------------------------------
    # ML-ready output
    # ---------------------------------------------------------

    def load_long_format(self) -> pd.DataFrame:
        """Returns long-format ML-ready dataset."""
        wide = self._load_snapshot()
        return wide_to_long(wide)


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run_today_ingestion()
