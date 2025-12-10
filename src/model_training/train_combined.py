# ============================================================
# File: src/model_training/train_combined.py
# Purpose: Unified training interface for LogReg or XGB
# Project: nba_analysis
# Version: 1.0
# ============================================================

from .train_logreg import train_logreg
from .train_xgb import train_xgb

def train_model(cache_file, out_dir="models", model_type="logreg"):
    if model_type=="logreg":
        return train_logreg(cache_file, out_dir)
    elif model_type=="xgb":
        return train_xgb(cache_file, out_dir)
    else:
        raise ValueError("Unknown model type")
