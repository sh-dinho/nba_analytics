#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incrementally update NBA master schedule
Author: Mohamadou
"""

import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def refresh_incremental(today_schedule, config):
    """
    Incrementally update the master schedule with today's schedule.
    """
    master_file = Path(config.schedule.master_file)
    if master_file.exists():
        master_schedule = pd.read_parquet(master_file)
    else:
        master_schedule = pd.DataFrame()

    master_schedule = pd.concat([master_schedule, today_schedule], ignore_index=True)
    master_schedule.to_parquet(master_file, index=False)
    logger.info(
        f"Incremental schedule updated and saved to {master_file} ({len(master_schedule)} rows)"
    )
    return master_schedule
