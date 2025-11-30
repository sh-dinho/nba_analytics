import os, time, logging, joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from data import fetch_historical_games, engineer_features
from db_module import record_model_metrics

logging.basicConfig(level=logging.INFO)

def ensure_model_dir():
    os.makedirs("models", exist_ok=True)

def is_cache_valid(path, ttl: int):
    return os.path.exists(path) and (time.time() - os.path.getmtime(path) < ttl)

def train_models(df_feat: pd.DataFrame):
    X = df_feat.drop(columns=["home_win", "total_points"])
    y_class = df_feat["home_win"]
    y_reg = df_feat["total_points"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train_c, y_train_c)

    reg = LinearRegression()
    reg.fit(X_train_r, y_train_r)

    y_pred_c = clf.predict(X_test_c)
    y_proba_c = clf.predict_proba(X_test_c)[:, 1]
    y_pred_r = reg.predict(X_test_r)

    acc = accuracy_score(y_test_c, y_pred_c)
    r2 = r2_score(y_test_r, y_pred_r)

    record_model_metrics(auc=None, accuracy=acc)
    logging.info(f"✔ Model trained. Accuracy={acc:.3f}, R²={r2:.3f}")

    eval_class = (X_test_c, y_test_c, y_pred_c, y_proba_c, acc)
    eval_reg = (y_test_r, y_pred_r, r2)
    return clf, reg, eval_class, eval_reg

def train_models_cached(season: str, ttl: int = 600):
    clf_path = f"models/clf_{season}.pkl"
    reg_path = f"models/reg_{season}.pkl"
    eval_class_path = f"models/eval_class_{season}.pkl"
    eval_reg_path = f"models/eval_reg_{season}.pkl"

    if all(is_cache_valid(p, ttl) for p in [clf_path, reg_path, eval_class_path, eval_reg_path]):
        try:
            clf = joblib.load(clf_path)
            reg = joblib.load(reg_path)
            eval_class = joblib.load(eval_class_path)
            eval_reg = joblib.load(eval_reg_path)
            logging.info("✔ Loaded models from cache")
            return clf, reg, eval_class, eval_reg
        except Exception as e:
            logging.warning(f"⚠️ Cache corrupted, retraining... {e}")

    df_games = fetch_historical_games(season)
    if df_games is None or df_games.empty:
        logging.error("❌ No historical games available")
        return None, None, None, None

    df_feat = engineer_features(df_games)
    clf, reg, eval_class, eval_reg = train_models(df_feat)

    ensure_model_dir()
    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)
    joblib.dump(eval_class, eval_class_path)
    joblib.dump(eval_reg, eval_reg_path)
    logging.info("✔ Models cached to disk")
    return clf, reg, eval_class, eval_reg