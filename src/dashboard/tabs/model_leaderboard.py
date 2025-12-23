from __future__ import annotations
# ============================================================
# 🏀 NBA Analytics v3
# Module: Dashboard — Model Leaderboard Tab
# File: src/dashboard/tabs/model_leaderboard.py
# Author: Sadiq
#
# Description:
#     Displays all trained models from the model registry
#     (index.json) in a sortable, filterable leaderboard.
#
#     Shows:
#       - model_name
#       - version
#       - feature_version
#       - train/test accuracy
#       - training date range
#       - created_at_utc
#       - production flag
#
#     Optional:
#       - Promote a model to production
# ============================================================


import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config.paths import MODEL_REGISTRY_DIR


def load_registry() -> pd.DataFrame:
    index_path = MODEL_REGISTRY_DIR / "index.json"
    if not index_path.exists():
        return pd.DataFrame()

    registry = json.loads(index_path.read_text())
    models = registry.get("models", [])

    if not models:
        return pd.DataFrame()

    return pd.DataFrame(models)


def promote_model(model_name: str, version: str):
    """
    Marks a specific model version as production=True
    and all others as production=False.
    """
    index_path = MODEL_REGISTRY_DIR / "index.json"
    registry = json.loads(index_path.read_text())

    for m in registry["models"]:
        if m["model_name"] == model_name and m["version"] == version:
            m["is_production"] = True
        else:
            m["is_production"] = False

    index_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def render_model_leaderboard():
    st.title("🏆 Model Leaderboard")
    st.caption("All trained models from the registry, sorted by creation date.")

    df = load_registry()

    if df.empty:
        st.warning("No models found in the registry.")
        return

    # Convert timestamps
    df["created_at_utc"] = pd.to_datetime(df["created_at_utc"])

    # Sort newest first
    df = df.sort_values("created_at_utc", ascending=False)

    # Display table
    st.subheader("📊 Model Registry Overview")
    st.dataframe(
        df[
            [
                "model_name",
                "version",
                "feature_version",
                "train_accuracy",
                "test_accuracy",
                "train_start_date",
                "train_end_date",
                "created_at_utc",
                "is_production",
            ]
        ],
        use_container_width=True,
    )

    # Production selection
    st.subheader("🚀 Promote Model to Production")

    model_names = df["model_name"].unique().tolist()
    selected_model = st.selectbox("Select model", model_names)

    versions = df[df["model_name"] == selected_model]["version"].tolist()
    selected_version = st.selectbox("Select version", versions)

    if st.button("Promote to Production"):
        promote_model(selected_model, selected_version)
        st.success(
            f"Model {selected_model} v{selected_version} promoted to production."
        )
        st.experimental_rerun()
