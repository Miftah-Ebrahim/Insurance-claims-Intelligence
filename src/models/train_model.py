import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    f1_score,
)
import joblib
import os

# Try importing XGBoost
try:
    from xgboost import XGBRegressor, XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. Skipping XGB models.")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelTrainer:
    """
    Class to train and evaluate Severity and Probability models.
    """

    def __init__(self):
        self.models = {}
        self.results = {}

    def train_severity_models(self, X_train, X_test, y_train, y_test):
        """
        Trains Regression models for Claim Severity.
        Target: TotalClaims
        """
        logging.info("Training Severity Models (Regression)...")

        # 1. Linear Regression (Baseline)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        self._evaluate_regression(lr, X_test, y_test, "LinearRegression")
        self.models["Severity_LR"] = lr

        # 2. Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        self._evaluate_regression(rf, X_test, y_test, "RandomForest_Reg")
        self.models["Severity_RF"] = rf

        # 3. XGBoost
        if XGB_AVAILABLE:
            xgb = XGBRegressor(
                n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1
            )
            xgb.fit(X_train, y_train)
            self._evaluate_regression(xgb, X_test, y_test, "XGBoost_Reg")
            self.models["Severity_XGB"] = xgb

    def train_probability_models(self, X_train, X_test, y_train, y_test):
        """
        Trains Classification models for Claim Probability.
        Target: IsClaim
        """
        logging.info("Training Probability Models (Classification)...")

        # Calculate scale_pos_weight for XGBoost (num_negative / num_positive)
        # Check if we have positive cases in train set to avoid div by zero
        num_pos = (y_train == 1).sum()
        num_neg = (y_train == 0).sum()
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

        # 1. Logistic Regression (Baseline) - Balanced features
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(X_train, y_train)
        self._evaluate_classification(lr, X_test, y_test, "LogisticRegression")
        self.models["Probability_LR"] = lr

        # 2. Random Forest - Balanced features
        rf = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
        )
        rf.fit(X_train, y_train)
        self._evaluate_classification(rf, X_test, y_test, "RandomForest_Clf")
        self.models["Probability_RF"] = rf

        # 3. XGBoost - Scaled weight
        if XGB_AVAILABLE:
            xgb = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
            )
            xgb.fit(X_train, y_train)
            self._evaluate_classification(xgb, X_test, y_test, "XGBoost_Clf")
            self.models["Probability_XGB"] = xgb

    def _evaluate_regression(self, model, X_test, y_test, name):
        """
        Calculates RMSE and R2 for regression models.
        """
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        logging.info(f"[{name}] RMSE: {rmse:.2f}, R2: {r2:.4f}")
        self.results[name] = {"RMSE": rmse, "R2": r2}

    def _evaluate_classification(self, model, X_test, y_test, name):
        """
        Calculates Accuracy and F1 for classification models.
        """
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        logging.info(f"[{name}] Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
        self.results[name] = {"Accuracy": acc, "F1": f1}

    def get_results(self):
        return pd.DataFrame(self.results).T

    def save_model(self, name, filepath):
        if name in self.models:
            joblib.dump(self.models[name], filepath)
            logging.info(f"Model saved to {filepath}")
        else:
            logging.error(f"Model {name} not found.")
