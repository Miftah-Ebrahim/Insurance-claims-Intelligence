import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataBuilder:
    """
    A class to handle data preprocessing, imputation, encoding, and splitting for insurance data.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.encoders = {}
        self.imputers = {}

    def preprocess(self):
        """
        Executes the full preprocessing pipeline.
        """
        logging.info("Starting preprocessing pipeline...")
        self._feature_engineering()  # Step 1: Create features (may introduce NaNs)
        self._handle_missing_values()  # Step 2: Clean everything (raw + new features)
        self._encode_categorical()  # Step 3: Encode
        logging.info("Preprocessing pipeline complete.")
        return self.df

    def _handle_missing_values(self):
        """
        Imputes missing values:
        - Numerical: Median
        - Categorical: Mode
        """
        logging.info("Handling missing values...")

        # Replace infinite values with NaN first
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Numerical
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if not num_cols.empty:
            # Check for columns that are all NaN to avoid error
            valid_num_cols = [c for c in num_cols if self.df[c].notna().sum() > 0]
            if valid_num_cols:
                imputer_num = SimpleImputer(strategy="median")
                self.df[valid_num_cols] = imputer_num.fit_transform(
                    self.df[valid_num_cols]
                )
                self.imputers["num"] = imputer_num

            # Fill remaining NaNs (e.g. all-NaN numeric columns) with 0
            if self.df[num_cols].isna().any().any():
                self.df[num_cols] = self.df[num_cols].fillna(0)

        # Categorical
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns
        if not cat_cols.empty:
            valid_cat_cols = [c for c in cat_cols if self.df[c].notna().sum() > 0]
            if valid_cat_cols:
                imputer_cat = SimpleImputer(strategy="most_frequent")
                self.df[valid_cat_cols] = imputer_cat.fit_transform(
                    self.df[valid_cat_cols]
                )
                self.imputers["cat"] = imputer_cat

            # Fill remaining NaNs (e.g. all-NaN categorical columns) with 'Unknown'
            if self.df[cat_cols].isna().any().any():
                self.df[cat_cols] = self.df[cat_cols].fillna("Unknown")

    def _feature_engineering(self):
        """
        Creates domain-specific features.
        """
        logging.info("Engineering features...")

        # 1. Claim Probability Target
        if "TotalClaims" in self.df.columns:
            self.df["IsClaim"] = (self.df["TotalClaims"] > 0).astype(int)

        # 2. Vehicle Age (Current Year - RegistrationYear)
        if "RegistrationYear" in self.df.columns:
            current_year = 2025  # Assuming current context or max year in data
            # Clean RegistrationYear (replace placeholders like 9999)
            self.df["RegistrationYear"] = pd.to_numeric(
                self.df["RegistrationYear"], errors="coerce"
            )
            self.df["VehicleAge"] = current_year - self.df["RegistrationYear"]
            # Handle anomalous ages
            self.df.loc[
                (self.df["VehicleAge"] < 0) | (self.df["VehicleAge"] > 50), "VehicleAge"
            ] = self.df["VehicleAge"].median()

        # 3. Premium to SumInsured Ratio (Proxy for Risk Rate)
        if "TotalPremium" in self.df.columns and "SumInsured" in self.df.columns:
            # SumInsured can be 0, so adding epsilon
            self.df["Premium_Risk_Ratio"] = self.df["TotalPremium"] / (
                self.df["SumInsured"] + 1e-6
            )

    def _encode_categorical(self):
        """
        Encodes categorical columns using Label Encoding (suitable for Trees/RF/XGB).
        """
        logging.info("Encoding categorical variables...")
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns

        for col in cat_cols:
            # Convert to string to ensure consistent type for encoder
            self.df[col] = self.df[col].astype(str)
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.encoders[col] = le

    def get_severity_data(self):
        """
        Returns X, y for Severity Model (only claims > 0).
        Target: TotalClaims
        """
        if "TotalClaims" not in self.df.columns:
            raise ValueError("TotalClaims column missing.")

        data = self.df[self.df["TotalClaims"] > 0].copy()
        y = data["TotalClaims"]
        X = data.drop(columns=["TotalClaims", "IsClaim"])  # Drop targets

        # Drop non-predictive ID columns if they exist (heuristic)
        cols_to_drop = [c for c in ["PolicyID", "Date"] if c in X.columns]
        X = X.drop(columns=cols_to_drop)

        return X, y

    def get_probability_data(self):
        """
        Returns X, y for Probability Model (all data).
        Target: IsClaim
        """
        if "IsClaim" not in self.df.columns:
            raise ValueError("IsClaim column missing. Run preprocessing first.")

        y = self.df["IsClaim"]
        X = self.df.drop(
            columns=["TotalClaims", "IsClaim"]
        )  # Drop targets and future info

        # Drop non-predictive ID columns
        cols_to_drop = [c for c in ["PolicyID", "Date"] if c in X.columns]
        X = X.drop(columns=cols_to_drop)

        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Wrapper for train_test_split.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
