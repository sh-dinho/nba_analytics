from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import joblib, os, logging
import pandas as pd

def train_xgb(cache_file, out_dir="models"):
    df = pd.read_parquet(cache_file)

    X = df.drop(columns=["target"], errors="ignore")
    y = df["target"] if "target" in df.columns else None
    if y is None:
        logging.error("Target column missing")
        return {"metrics": {"logloss": None}, "model_path": None}

    # Separate numeric and categorical features
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
    return {"metrics": {"logloss": loss}, "model_path": model_path}
