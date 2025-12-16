# ============================================================
# File: run_pipeline_offline.py
# Purpose: NBA pipeline with incremental updates & offline fallback
# ============================================================

import logging
from pathlib import Path
import pandas as pd
import pickle
from datetime import datetime
import requests
from sklearn.ensemble import RandomForestClassifier

# ================== CONFIG ==================
DATA_RAW = Path("data/raw")
DATA_CACHE = Path("data/cache")
DATA_HISTORY = Path("data/history")
MODEL_PATH = Path("models/nba_model.pkl")
SEASONS = [2022, 2023, 2024, 2025]
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# ================== HELPERS ==================


def fetch_season_schedule(year: int) -> pd.DataFrame:
    """Fetch season schedule from NBA API or fallback to cached JSON."""
    json_path = DATA_RAW / f"schedule_{year}.json"
    # Attempt to load local cache first
    if json_path.exists():
        try:
            df = pd.read_json(json_path)
            logging.info(f"Loaded cached JSON for season {year} ({len(df)} rows)")
            return df
        except Exception as e:
            logging.warning(f"Failed to load cached JSON for {year}: {e}")
    # Try API fetch
    url = f"https://data.nba.net/prod/v2/{year}/schedule.json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        games = data.get("league", {}).get("standard", [])
        df = pd.DataFrame(games)
        df.to_json(json_path, orient="records", lines=False)
        logging.info(f"Fetched and cached season {year} ({len(df)} rows)")
        return df
    except Exception as e:
        logging.error(f"Failed to fetch season {year}: {e}")
        # Fallback: try loading historical parquet
        parquet_path = DATA_HISTORY / f"schedule_{year}.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            logging.info(f"Loaded fallback parquet for season {year} ({len(df)} rows)")
            return df
        return pd.DataFrame()


def load_historical_schedule() -> pd.DataFrame:
    """Load combined historical schedule."""
    if not DATA_HISTORY.exists():
        DATA_HISTORY.mkdir(parents=True, exist_ok=True)
    files = list(DATA_HISTORY.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Historical schedule loaded (rows={len(df)})")
    return df


def save_historical_schedule(df: pd.DataFrame):
    """Save historical schedule per season."""
    for season in df["seasonYear"].unique():
        season_df = df[df["seasonYear"] == season]
        path = DATA_HISTORY / f"schedule_{season}.parquet"
        season_df.to_parquet(path, index=False)
        logging.info(f"Saved Parquet for {season} ({len(season_df)} rows)")


def incremental_update(historical: pd.DataFrame) -> pd.DataFrame:
    """Fetch only missing games incrementally."""
    new_data = []
    for season in SEASONS:
        season_df = (
            historical[historical["seasonYear"] == season]
            if not historical.empty
            else pd.DataFrame()
        )
        last_date = (
            pd.to_datetime(season_df["startDateEastern"]).max()
            if not season_df.empty
            else None
        )

        df_season = fetch_season_schedule(season)
        if df_season.empty:
            continue
        df_season["startDateEastern"] = pd.to_datetime(df_season["startDateEastern"])
        if last_date:
            df_season = df_season[df_season["startDateEastern"] > last_date]

        if not df_season.empty:
            new_data.append(df_season)

    if new_data:
        incremental_df = pd.concat(new_data, ignore_index=True)
        updated_historical = pd.concat([historical, incremental_df], ignore_index=True)
        save_historical_schedule(updated_historical)
        logging.info(f"Incremental update: added {len(incremental_df)} new rows")
        return updated_historical
    else:
        logging.info("No new games to update.")
        return historical


# ================== FEATURE ENGINEERING ==================


def prepare_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    if schedule.empty:
        logging.warning("Schedule is empty; returning empty features DataFrame.")
        return pd.DataFrame()
    df = schedule.copy()
    df["WIN_STREAK"] = 0
    df["POINT_DIFF"] = df.get("homeScore", 0) - df.get("awayScore", 0)
    feature_cols = ["WIN_STREAK", "POINT_DIFF"]
    df_features = df[feature_cols].copy()
    df_features["game_id"] = df.get("gameId", range(len(df)))
    return df_features


# ================== MODEL ==================


def train_model(historical_schedule: pd.DataFrame):
    feature_cols = ["WIN_STREAK", "POINT_DIFF"]
    X = historical_schedule[feature_cols]
    y = historical_schedule.get("outcome", pd.Series([0] * len(X)))
    if y.nunique() < 2:
        raise ValueError("Not enough classes to train model.")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def load_model(model_path: Path, historical_schedule: pd.DataFrame):
    auto_trained = False
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {model_path}")
    else:
        logging.warning("Model not found; training a new one...")
        try:
            model = train_model(historical_schedule)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            auto_trained = True
            logging.info(f"New model trained and saved to {model_path}")
        except Exception as e:
            logging.warning(f"Failed to train model: {e}")
            model = None
    return model, auto_trained


# ================== PREDICTIONS & RANKINGS ==================


def predict_games(features: pd.DataFrame, model):
    if features.empty or model is None:
        return pd.DataFrame()
    feature_cols = ["WIN_STREAK", "POINT_DIFF"]
    features["predicted_win"] = model.predict_proba(features[feature_cols])[:, 1]
    features["predicted_outcome"] = (features["predicted_win"] > 0.5).astype(int)
    return features


def generate_rankings(predictions: pd.DataFrame):
    if predictions.empty:
        return pd.DataFrame()
    predictions["rank"] = predictions["predicted_win"].rank(
        method="min", ascending=False
    )
    predictions["bet_signal"] = (predictions["predicted_win"] > 0.6).astype(int)
    return predictions.sort_values("rank").reset_index(drop=True)


# ================== MAIN ==================


def main():
    logging.info("===== NBA DAILY PIPELINE START =====")

    for folder in [DATA_RAW, DATA_CACHE, DATA_HISTORY]:
        folder.mkdir(parents=True, exist_ok=True)

    historical = load_historical_schedule()

    if historical.empty:
        logging.info("No historical data; downloading full seasons.")
        historical = pd.concat(
            [
                fetch_season_schedule(season)
                for season in SEASONS
                if not fetch_season_schedule(season).empty
            ],
            ignore_index=True,
        )
        save_historical_schedule(historical)
    else:
        logging.info("Historical data exists; checking for updates.")
        historical = incremental_update(historical)

    features = prepare_schedule(historical)
    logging.info("Features prepared for ML model")

    model, auto_trained = load_model(MODEL_PATH, historical)
    predictions = predict_games(features, model)
    rankings = generate_rankings(predictions)

    if not rankings.empty:
        logging.info("Rankings & betting signals ready")
    else:
        logging.warning("No rankings generated.")

    logging.info("===== NBA DAILY PIPELINE END =====")


if __name__ == "__main__":
    main()
