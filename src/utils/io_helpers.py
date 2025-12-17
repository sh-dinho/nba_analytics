import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def load_master_schedule(file_path: Path) -> pd.DataFrame:
    if file_path.exists():
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded master schedule from {file_path} (rows={len(df)})")
            return df
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return pd.DataFrame()
    logger.warning(f"Master schedule not found at {file_path}")
    return pd.DataFrame()


def save_master_schedule(df: pd.DataFrame, file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df["last_saved"] = datetime.now().isoformat()
        df.to_parquet(file_path, index=False)
        logger.info(f"Saved master schedule to {file_path} (rows={len(df)})")
    except Exception as e:
        logger.error(f"Failed to save {file_path}: {e}")
        raise
