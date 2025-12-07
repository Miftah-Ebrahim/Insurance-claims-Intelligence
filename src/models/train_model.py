import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.loader import load_data
from src.features.preprocessing import (
    get_preprocessor,
    get_feature_columns,
    get_target_column,
)

# Constants
DATA_PATH = os.path.join("data", "raw", "machine_learning_rating.txt")
MODEL_PATH = os.path.join("models", "random_forest_v1.pkl")
RANDOM_STATE = 42


def train_pipeline():
    print("Loading data...")
    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    # 1. Feature Selection
    target_col = get_target_column()
    feature_cols = get_feature_columns()

    numeric_features = feature_cols["numeric"]
    categorical_features = feature_cols["categorical"]

    # Check if columns exist
    available_cols = df.columns
    numeric_features = [c for c in numeric_features if c in available_cols]
    categorical_features = [c for c in categorical_features if c in available_cols]

    print(
        f"Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical"
    )

    X = df[numeric_features + categorical_features]
    y = df[target_col]

    # Handle missing targets if any (drop rows)
    if y.isnull().any():
        print(f"Dropping {y.isnull().sum()} rows with missing target values...")
        # Align X and y
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]

    # 2. Train/Test Split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # 3. Build Pipeline
    print("Building pipeline...")
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1
                ),
            ),
            # n_estimators=50 for speed in demo, n_jobs=-1 for parallel
        ]
    )

    # 4. Train
    print("Training model (this may take a while)...")
    pipeline.fit(X_train, y_train)
    print("Model trained.")

    # 5. Evaluate
    print("Evaluating...")
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\n--- Model Performance ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    # 6. Save Model
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_pipeline()
