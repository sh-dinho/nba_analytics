from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Training-Time Feature Schema Saver
# File: src/model/schema.py
# Author: Sadiq
#
# Description:
#     Saves and loads training-time feature schemas, including:
#       - Feature list (ordered)
#       - Dtypes
#       - Min/max
#       - Mean/std
#       - Missing-value counts
#       - Training timestamp
#       - Model version
#
#     Used to enforce strict schema alignment at prediction time.
# ============================================================

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from loguru import logger


# ------------------------------------------------------------
# Schema dataclass
# ------------------------------------------------------------


@dataclass
class FeatureSchema:
    model_type: str
    model_version: str
    feature_version: str
    created_at: str
    feature_cols: list[str]
    dtypes: dict
    min: dict
    max: dict
    mean: dict
    std: dict
    missing: dict

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------
# Schema Saver
# ------------------------------------------------------------


class SchemaSaver:
    """
    Saves training-time schema for a model.
    """

    def __init__(self, model_type: str, model_version: str, feature_version: str):
        self.model_type = model_type
        self.model_version = model_version
        self.feature_version = feature_version

    def save(self, X: pd.DataFrame, out_path: Path) -> None:
        """
        Save schema for the training feature matrix X.
        """

        schema = FeatureSchema(
            model_type=self.model_type,
            model_version=self.model_version,
            feature_version=self.feature_version,
            created_at=datetime.utcnow().isoformat(),
            feature_cols=list(X.columns),
            dtypes={c: str(X[c].dtype) for c in X.columns},
            min={c: float(np.nanmin(X[c])) for c in X.columns},
            max={c: float(np.nanmax(X[c])) for c in X.columns},
            mean={c: float(np.nanmean(X[c])) for c in X.columns},
            std={c: float(np.nanstd(X[c])) for c in X.columns},
            missing={c: int(X[c].isna().sum()) for c in X.columns},
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(schema.to_dict(), f, indent=2)

        logger.success(f"[SchemaSaver] Saved training schema → {out_path}")


# ------------------------------------------------------------
# Schema Loader
# ------------------------------------------------------------


class SchemaLoader:
    """
    Loads a saved schema and provides validation utilities.
    """

    def __init__(self, schema_path: Path):
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with schema_path.open("r", encoding="utf-8") as f:
            self.schema = json.load(f)

        self.feature_cols = self.schema["feature_cols"]
        self.dtypes = self.schema["dtypes"]
        self.min = self.schema["min"]
        self.max = self.schema["max"]
        self.mean = self.schema["mean"]
        self.std = self.schema["std"]
        self.missing = self.schema["missing"]

    # --------------------------------------------------------
    # Validation helpers
    # --------------------------------------------------------

    def validate_columns(self, X: pd.DataFrame) -> dict:
        """
        Validate column presence and return a report.
        """
        missing = [c for c in self.feature_cols if c not in X.columns]
        extra = [c for c in X.columns if c not in self.feature_cols]

        return {
            "missing_columns": missing,
            "extra_columns": extra,
            "is_valid": len(missing) == 0,
        }

    def validate_dtypes(self, X: pd.DataFrame) -> dict:
        """
        Validate dtype consistency.
        """
        mismatches = {}
        for c in self.feature_cols:
            if c in X.columns:
                if str(X[c].dtype) != self.dtypes[c]:
                    mismatches[c] = {
                        "expected": self.dtypes[c],
                        "actual": str(X[c].dtype),
                    }

        return {
            "dtype_mismatches": mismatches,
            "is_valid": len(mismatches) == 0,
        }

    def detect_drift(self, X: pd.DataFrame) -> dict:
        """
        Detect distribution drift using min/max/mean/std.
        """
        drift = {}

        for c in self.feature_cols:
            if c not in X.columns:
                continue

            col = X[c]

            drift[c] = {
                "min_diff": float(col.min() - self.min[c]),
                "max_diff": float(col.max() - self.max[c]),
                "mean_diff": float(col.mean() - self.mean[c]),
                "std_diff": float(col.std() - self.std[c]),
            }

        return drift
