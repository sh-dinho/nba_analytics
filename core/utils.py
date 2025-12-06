# ============================================================
# File: core/utils.py
# Purpose: Utility functions for NBA analytics project
# ============================================================

import pandas as pd

def convert_object_to_category(df: pd.DataFrame, exclude: list = None) -> pd.DataFrame:
    """
    Convert all object columns in a DataFrame to category dtype, except those in `exclude`.

    Args:
        df (pd.DataFrame): Input DataFrame
        exclude (list, optional): List of column names to skip conversion. Defaults to None.

    Returns:
        pd.DataFrame: Updated DataFrame with categorical columns
    """
    if exclude is None:
        exclude = []

    for col in df.select_dtypes(include="object").columns:
        if col not in exclude:
            df[col] = df[col].astype("category")
    return df
