import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def get_preprocessor(numeric_features: list, categorical_features: list):
    """
    Creates a Scikit-Learn ColumnTransformer for preprocessing.

    Args:
        numeric_features (list): List of numeric column names.
        categorical_features (list): List of categorical column names.

    Returns:
        ColumnTransformer: Configured preprocessor.
    """

    # Numeric pipeline: Impute with median
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    # Categorical pipeline: Impute with 'missing' and OneHotEncode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def get_target_column():
    return "TotalClaims"


def get_feature_columns():
    return {
        "numeric": [
            "TotalPremium",
            "SumInsured",
            "CalculatedPremiumPerTerm",
        ],  # Example numeric features
        "categorical": [
            "Gender",
            "VehicleType",
            "MaritalStatus",
            "Province",
        ],  # Example categorical features
    }
