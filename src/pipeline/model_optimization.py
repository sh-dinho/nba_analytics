import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.impute import SimpleImputer  # For handling missing values
from loguru import logger
from typing import List


def _select_training_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """
    Select model input features by excluding identifiers and target.
    """
    exclude = {"game_id", "date", "home_team", "away_team", target_col}
    return [c for c in df.columns if c not in exclude]


def train_and_evaluate_models(df, features, target):
    """
    Train the model with handling class imbalance using:
        1. SMOTE (oversampling minority class)
        2. RandomUnderSampler (undersampling majority class)
        3. Class Weights in RandomForestClassifier
    It also handles the creation of the 'home_minus_away' feature if it doesn't exist.
    """

    logger.info("Starting model training and evaluation")

    # Check if the 'home_minus_away' column exists
    logger.info(f"Columns in the DataFrame: {df.columns}")

    if "home_minus_away" not in df.columns:
        logger.warning("'home_minus_away' column is missing, creating it...")
        df["home_minus_away"] = df["home_score"] - df["away_score"]

    # Check if there are missing values in the 'home_score' or 'away_score' columns
    missing_scores = df[["home_score", "away_score"]].isnull().sum()
    if missing_scores.any():
        logger.warning(f"Missing values in the score columns: {missing_scores}")

    # Create the 'homeWin' target column if it doesn't exist
    if "homeWin" not in df.columns:
        logger.warning("'homeWin' column is missing, creating it...")
        df["homeWin"] = (df["home_score"] > df["away_score"]).astype(int)

    # Handling missing values
    logger.info("Handling missing values (if any) using imputation...")

    # Ensure columns that we are imputing have valid data (i.e., avoid imputing newly created columns)
    impute_columns = [
        col for col in features if col in df.columns and df[col].isnull().any()
    ]

    logger.info(f"Columns with missing values: {impute_columns}")

    if impute_columns:
        # We can skip columns that have all NaN values
        valid_columns = [
            col for col in impute_columns if df[col].isnull().sum() < len(df)
        ]

        logger.info(f"Valid columns for imputation: {valid_columns}")

        if valid_columns:
            logger.info(f"Imputing missing values in columns: {valid_columns}")
            imputer = SimpleImputer(strategy="mean")
            df[valid_columns] = imputer.fit_transform(df[valid_columns])
        else:
            logger.warning(
                "No valid columns to impute. All specified columns contain only NaN values."
            )

        # Check if columns are still NaN after imputation
        if df[valid_columns].isnull().sum().sum() > 0:
            logger.error(
                "There are still NaN values in the input features after imputation."
            )
            return None  # Stop execution if there are still NaNs in the data
    else:
        logger.info("No missing values to impute in the selected features.")

    # Verify if any features still contain NaN values
    if df[features].isnull().sum().sum() > 0:
        logger.warning("Some features still contain NaN values after imputation.")
        # Optionally, drop rows with NaNs or replace NaNs
        df = df.dropna(subset=features)  # Drop rows with NaNs in the features
        logger.info(f"Dropped rows with NaNs. Remaining rows: {len(df)}")

    # If the dataset is empty after dropping NaNs, stop further execution
    if df.empty:
        logger.error("Dataset is empty after removing NaNs. Aborting model training.")
        return None

    # Proceed with feature selection
    X = df[features]
    y = df[target]

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Apply SMOTE for oversampling
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Apply Random Under-sampling
    undersample = RandomUnderSampler(sampling_strategy="auto", random_state=42)
    X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)

    # Create Random Forest model with class weights
    model = RandomForestClassifier(class_weight="balanced", random_state=42)

    # Evaluate with both resampling techniques
    resampling_techniques = [
        ("SMOTE", X_train_smote, y_train_smote),
        ("Undersampling", X_train_under, y_train_under),
    ]

    for technique, X_train_resampled, y_train_resampled in resampling_techniques:
        # Train model
        model.fit(X_train_resampled, y_train_resampled)

        # Evaluate model on the test set
        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        y_pred_proba = model.predict_proba(X_test)[
            :, 1
        ]  # Get probabilities for ROC-AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Log the results
        logger.info(f"{technique} - Model Evaluation Metrics:")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"ROC-AUC: {roc_auc:.4f}")

    return model


# ---------------------------------------------------------
# Helpers for prediction pipeline
# ---------------------------------------------------------


def load_model():
    """
    Load the trained model from the model registry (or disk).
    """
    from src.model.registry import ModelRegistry

    registry = ModelRegistry()
    model, metadata, version = registry.load_latest()
    if model is None:
        raise FileNotFoundError("No trained model found in registry.")
    return model


def load_metadata():
    """
    Load the metadata of the latest trained model from the model registry.
    """
    from src.model.registry import ModelRegistry

    registry = ModelRegistry()
    model, metadata, version = registry.load_latest()
    if metadata is None:
        raise FileNotFoundError("No metadata found for the latest model.")
    return metadata


# ---------------------------------------------------------
# Cross-validation to assess model performance
# ---------------------------------------------------------


def cross_val_metrics(model, X, y):
    """
    Evaluate model using cross-validation with different metrics.
    """
    # Cross-validation with Precision as scoring metric
    cv_precision = cross_val_score(
        model, X, y, cv=5, scoring=make_scorer(precision_score)
    )
    logger.info(f"Cross-validation precision scores: {cv_precision}")
    logger.info(f"Average precision score from CV: {cv_precision.mean():.4f}")

    # Similarly, you can use recall_score, f1_score, etc., for cross-validation.
    cv_recall = cross_val_score(model, X, y, cv=5, scoring=make_scorer(recall_score))
    logger.info(f"Cross-validation recall scores: {cv_recall}")
    logger.info(f"Average recall score from CV: {cv_recall.mean():.4f}")

    cv_f1 = cross_val_score(model, X, y, cv=5, scoring=make_scorer(f1_score))
    logger.info(f"Cross-validation F1 scores: {cv_f1}")
    logger.info(f"Average F1 score from CV: {cv_f1.mean():.4f}")

    cv_roc_auc = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    logger.info(f"Cross-validation ROC-AUC scores: {cv_roc_auc}")
    logger.info(f"Average ROC-AUC score from CV: {cv_roc_auc.mean():.4f}")
