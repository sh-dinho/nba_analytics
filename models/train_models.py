import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Assuming Streamlit is used for the front-end, we can use its caching mechanism
# import streamlit as st 

# Import data preparation functions from the core package
from nba_analytics_core.data import fetch_historical_games, engineer_features, current_season

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# NOTE: In a real Streamlit app, you would decorate this function:
# @st.cache_data(ttl=24 * 3600) # Cache for 24 hours
def train_models_cached(
    seasons: list[str] = None, 
    model_path: str = "artifacts/classification_model.pkl"
) -> dict:
    """
    Fetches historical data, trains a Random Forest Classifier, saves it,
    and returns a summary of the training results.
    """
    if seasons is None:
        # Default to the last 5 seasons including the current one
        current = current_season()
        start_year = int(current.split('-')[0])
        seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(start_year - 4, start_year + 1)]
    
    logging.info(f"Starting model training for seasons: {seasons}")
    games = fetch_historical_games(seasons)
    
    if games.empty:
        logging.error("Cannot train model: No historical games were fetched.")
        return {"status": "failed", "message": "No data for training."}

    X, y = engineer_features(games)
    
    if X.empty or y.empty:
        logging.error("Cannot train model: Feature engineering returned empty sets.")
        return {"status": "failed", "message": "Empty features/labels."}

    # 1. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logging.info(f"Training set size: {len(X_train)} | Test set size: {len(X_test)}")

    # 2. Train Model
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 3. Evaluate Model
    acc = model.score(X_test, y_test)
    logging.info(f"Model accuracy on test set: {acc:.4f}")
    
    # Additional metric (e.g., F1 Score or AUC) would be added here
    # Example: from sklearn.metrics import f1_score
    # f1 = f1_score(y_test, model.predict(X_test))

    # 4. Save Model Artifact
    joblib.dump(model, model_path)
    logging.info(f"✔ Model saved to {model_path}")
    
    return {
        "status": "success",
        "accuracy": float(f"{acc:.4f}"),
        "training_time": datetime.now().isoformat(),
        "features": list(X.columns)
    }

if __name__ == "__main__":
    # Example usage: train model on hardcoded list of seasons
    current = current_season()
    seasons_to_train = ["2021-22", "2022-23", "2023-24", "2024-25", current]
    results = train_models_cached(seasons=seasons_to_train)
    print("\nTraining Summary:")
    print(pd.Series(results).to_markdown())