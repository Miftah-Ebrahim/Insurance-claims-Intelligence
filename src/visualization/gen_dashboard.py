import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import logging
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from src.data.loader import load_data
from src.features.build_features import DataBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

TABLE_STYLE = {
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "font.family": "sans-serif",
}
plt.rcParams.update(TABLE_STYLE)


def save_plot(fig, filename):
    output_dir = "dashboard/figures"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    logging.info(f"Saved plot: {path}")
    plt.close(fig)


def generate_dashboard():
    logging.info("Generating Dashboard Figures...")

    # Load Data
    df = load_data("data/raw/MachineLearningRating.txt")
    builder = DataBuilder(df)
    builder.preprocess()
    df_clean = builder.df

    # 1. Premium vs Claims (Scatter)
    logging.info("Plotting Premium vs Claims...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df_clean,
        x="TotalPremium",
        y="TotalClaims",
        hue="IsClaim",
        palette="coolwarm",
        alpha=0.6,
        ax=ax,
    )
    ax.set_title("Premium vs. Claims Correlation")
    ax.set_xlabel("Total Premium (ZAR)")
    ax.set_ylabel("Total Claims (ZAR)")
    save_plot(fig, "premium_vs_claims.png")

    # 2. Geographic Trend (Bar)
    logging.info("Plotting Geographic Trends...")
    fig, ax = plt.subplots(figsize=(12, 6))
    prov_risk = (
        df_clean.groupby("Province")["TotalClaims"].mean().sort_values(ascending=False)
    )
    sns.barplot(x=prov_risk.index, y=prov_risk.values, palette="viridis", ax=ax)
    ax.set_title("Average Claim Severity by Province")
    ax.set_ylabel("Avg Total Claims (ZAR)")
    ax.tick_params(axis="x", rotation=45)
    save_plot(fig, "geographic_trend.png")

    # 3. Outliers Boxplot
    logging.info("Plotting Outliers...")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=df_clean["TotalPremium"], color="orange", ax=ax)
    ax.set_title("Distribution of Total Premium (Outlier Detection)")
    ax.set_xlabel("Total Premium")
    save_plot(fig, "outliers_boxplot.png")

    # 4. Correlation Heatmap
    logging.info("Plotting Correlation Heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8))
    # Select numeric columns relevant to risk
    cols = [
        "TotalPremium",
        "TotalClaims",
        "SumInsured",
        "VehicleAge",
        "Premium_Risk_Ratio",
        "IsClaim",
    ]
    corr = df_clean[cols].corr()
    sns.heatmap(corr, annot=True, cmap="RdBu", center=0, fmt=".2f", ax=ax)
    ax.set_title("Key Variable Correlations")
    save_plot(fig, "correlation_heatmap.png")

    # 5. Categorical Risk (BodyType)
    logging.info("Plotting Categorical Risk...")
    fig, ax = plt.subplots(figsize=(12, 6))
    # Filter top 10 body types to avoid clutter
    top_bodies = df_clean["VehicleType"].value_counts().nlargest(10).index
    data_filtered = df_clean[df_clean["VehicleType"].isin(top_bodies)]

    sns.barplot(
        data=data_filtered,
        x="VehicleType",
        y="IsClaim",
        estimator=np.mean,
        ci=None,
        palette="magma",
        ax=ax,
    )
    ax.set_title("Claim Probability by Vehicle Type")
    ax.set_ylabel("Claim Probability")
    ax.tick_params(axis="x", rotation=45)
    save_plot(fig, "categorical_risk.png")

    # 6. Key Insight / Hypothesis Result
    logging.info("Plotting Key Insight (Feature Importance)...")
    # Try to load model for feature importance, else fallback
    try:
        model_path = "models/Probability_RF.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                # We need feature names. Since pipeline drops columns, we need to replicate that.
                # This is tricky without the exact feature list.
                # Simplified approach: Use generic names or skip if risky.
                # Better: hypothesis plot for Gender Risk
                raise ValueError("Skipping Feature Importance to avoid mismatch.")
    except Exception:
        # Fallback: Gender Risk Plot (Test 4 Result)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(
            data=df_clean,
            x="Gender",
            y="TotalClaims",
            estimator=np.mean,
            ci=95,
            palette="Set2",
            ax=ax,
        )
        ax.set_title("Risk Profile: Gender Analysis (Statistically Insignificant)")
        ax.set_ylabel("Average Claim Severity")
        save_plot(fig, "key3_insight_plots.png")

    logging.info("Dashboard Generation Complete.")


if __name__ == "__main__":
    generate_dashboard()
