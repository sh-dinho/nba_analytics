# ============================================================
# Path: run_pipeline.py
# Purpose: End-to-end NBA analytics pipeline for current season
# Version: 3.0 (unified, automated, headers everywhere)
# ============================================================

import os
import logging
import datetime
import pandas as pd
import yaml
import mlflow
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# ============================================================
# Logging + Config
# ============================================================

def configure_logging(log_file="logs/pipeline.log"):
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8"),
                  logging.StreamHandler()],
    )
    logging.info("Logging configured. Writing to %s", log_file)

def load_config(config_file="config.yaml"):
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    return {"model_type": "logreg"}

# ============================================================
# Feature Generation
# ============================================================

def generate_features_for_games(game_data_list):
    features = []
    for i, game in enumerate(game_data_list):
        df = pd.DataFrame([game])
        if "GAME_ID" not in df.columns:
            df["GAME_ID"] = f"unknown_game_{i}"
        if "TEAM_ID" not in df.columns:
            df["TEAM_ID"] = -1
        features.append(df)
    valid_features = [df for df in features if not df.empty]
    return pd.concat(valid_features, ignore_index=True) if valid_features else pd.DataFrame()

def add_unique_id(df):
    if "GAME_ID" not in df.columns:
        df["GAME_ID"] = [f"unknown_game_{i}" for i in range(len(df))]
    if "TEAM_ID" not in df.columns:
        df["TEAM_ID"] = -1
    if "prediction_date" not in df.columns:
        df["prediction_date"] = datetime.date.today().isoformat()
    df["unique_id"] = (
        df["GAME_ID"].astype(str)
        + "_"
        + df["TEAM_ID"].astype(str)
        + "_"
        + df["prediction_date"].astype(str)
    )
    return df

def fetch_season_games(season):
    # Stub: replace with real API fetch
    games = []
    for i in range(50):  # demo: 50 games
        games.append({
            "GAME_ID": f"{season}{i:03d}",
            "TEAM_ID": i % 30,
            "Date": f"{season}-11-{(i % 28) + 1}",
            "HomeTeam": f"Team{(i % 15)}",
            "AwayTeam": f"Team{(i % 15)+1}",
            "points": 90 + (i % 30),
            "target": (i % 2)  # dummy win/loss
        })
    return games

def generate_current_season():
    current_year = datetime.date.today().year
    raw_games = fetch_season_games(season=current_year)
    features_df = generate_features_for_games(raw_games)
    features_df = add_unique_id(features_df)
    features_df["Season"] = current_year
    return features_df

# ============================================================
# Training Functions
# ============================================================

def train_logreg(cache_file, out_dir="models"):
    df = pd.read_parquet(cache_file)
    X = df.drop(columns=["target"], errors="ignore")
    y = df["target"] if "target" in df.columns else None
    if y is None:
        logging.error("Target column missing")
        return {"metrics": {"accuracy": None}, "model_path": None}

    numeric_features = X.select_dtypes(include=["number"]).columns
    categorical_features = X.select_dtypes(exclude=["number"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipeline.predict(X_test))

    os.makedirs(out_dir, exist_ok=True)
    model_path = f"{out_dir}/logreg.pkl"
    joblib.dump(pipeline, model_path)

    logging.info("Logistic Regression accuracy: %.3f", acc)
    return {"metrics": {"accuracy": acc}, "model_path": model_path, "features": list(X.columns)}

def train_xgb(cache_file, out_dir="models"):
    df = pd.read_parquet(cache_file)
    X = df.drop(columns=["target"], errors="ignore")
    y = df["target"] if "target" in df.columns else None
    if y is None:
        logging.error("Target column missing")
        return {"metrics": {"logloss": None}, "model_path": None}

    numeric_features = X.select_dtypes(include=["number"]).columns
    categorical_features = X.select_dtypes(exclude=["number"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, use_label_encoder=False,
            eval_metric="logloss"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    loss = log_loss(y_test, pipeline.predict_proba(X_test))

    os.makedirs(out_dir, exist_ok=True)
    model_path = f"{out_dir}/nba_xgb.pkl"
    joblib.dump(pipeline, model_path)

    logging.info("XGBoost logloss: %.3f", loss)
    return {"metrics": {"logloss": loss}, "model_path": model_path, "features": list(X.columns)}

# ============================================================
# Tracker + Visualization
# ============================================================

def build_game_tracker(features_df, predictions_df, player_info_df, used_features=None):
    tracker = features_df.merge(predictions_df, on="GAME_ID", how="left")
    tracker = tracker.merge(player_info_df, on=["GAME_ID", "TEAM_ID"], how="left")

    def classify(row):
        if row["prediction_confidence"] >= 0.75:
            return "Stake"
        elif row["prediction_confidence"] <= 0.55:
            return "Avoid"
        else:
            return "Watch"

    tracker["Recommendation"] = tracker.apply(classify, axis=1)

    if used_features is not None:
        tracker["FeaturesUsed"] = ", ".join(used_features)

    return tracker[["Season", "GAME_ID", "Date", "HomeTeam", "AwayTeam", "PlayerNames", "Recommendation", "FeaturesUsed"]]

def save_summary_chart(tracker, out_path):
    counts = tracker["Recommendation"].value_counts()
    plt.figure(figsize=(6,4))
    counts.plot(kind="bar", color=["green","red","blue"])
    plt.title("Recommendation Summary (Current Season)")
    plt.ylabel("Number of Games")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ============================================================
# Main Pipeline
# ============================================================

def main():
    configure_logging()
    config = load_config()
    model_type = config.get("model_type", "logreg")
    cache_file = "data/cache/features_full.parquet"

    logging.info("Pipeline started with model_type=%s", model_type)

    # Step 1: Generate features for current season
    features_df = generate_current_season()
    os.makedirs("data/cache", exist_ok=True)
    features_df.to_parquet(cache_file, index=False)
    logging.info("Cache regenerated for season %s with %d rows", features_df["Season"].iloc[0], len(features_df))

    # Step 2: Train model
    with mlflow.start_run(run_name=f"{model_type}_pipeline_v3.0"):
        if model_type == "logreg":
            result = train_logreg(cache_file=cache_file, out_dir="models")
            mlflow.log_metric("accuracy", result["metrics"]["accuracy"])
        elif model_type == "xgb":
            result = train_xgb(cache_file=cache_file, out_dir="models")
            mlflow.log_metric("logloss", result["metrics"]["logloss"])

            # Step 3: Build tracker
        predictions_df = pd.DataFrame({
            "GAME_ID": features_df["GAME_ID"],
            "prediction_confidence": 0.65  # demo confidence
        })
        player_info_df = pd.DataFrame({
            "GAME_ID": features_df["GAME_ID"],
            "TEAM_ID": features_df["TEAM_ID"],
            "PlayerNames": ["LeBron James"] * len(features_df)  # demo player info
        })

        tracker = build_game_tracker(
            features_df,
            predictions_df,
            player_info_df,
            used_features=result.get("features")
        )

        os.makedirs("data/tracker", exist_ok=True)
        tracker_csv = "data/tracker/games_tracker.csv"
        tracker.to_csv(tracker_csv, index=False)
        mlflow.log_artifact(tracker_csv, artifact_path="tracker")

        # Step 4: Save summary chart
        summary_chart_path = "data/tracker/recommendation_summary.png"
        save_summary_chart(tracker, summary_chart_path)
        mlflow.log_artifact(summary_chart_path, artifact_path="tracker")

        logging.info("Game tracker and summary chart saved and logged to MLflow")

    logging.info("Pipeline finished successfully")

    # ============================================================
    # Entry Point
    # ============================================================

    if __name__ == "__main__":
        main()
