from __future__ import annotations
from src.app.utils.pipeline_trigger import trigger_full_pipeline

def main():
    result = trigger_full_pipeline(
        backfill_days=0,
        skip_ingestion=False,
        skip_predictions=False,
    )
    print(result)

if __name__ == "__main__":
    main()