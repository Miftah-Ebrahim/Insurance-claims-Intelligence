import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_outliers_boxplots(df: pd.DataFrame, columns: list):
    """
    Generates boxplots for outlier detection.
    """
    valid_cols = [c for c in columns if c in df.columns]

    if not valid_cols:
        print("No valid columns found for boxplots.")
        return

    n_cols = 2
    n_rows = (len(valid_cols) + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(valid_cols):
        sns.boxplot(x=df[col].dropna(), ax=axes[i], color="skyblue")
        axes[i].set_title(f"Boxplot of {col}")
        axes[i].set_xlabel(col)

    for i in range(len(valid_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_geo_trends(df: pd.DataFrame, geo_col: str, value_col: str, title: str):
    """
    Bar plot for geographic comparison.
    """
    if geo_col not in df.columns or value_col not in df.columns:
        return

    plt.figure(figsize=(12, 6))
    data = df.groupby(geo_col)[value_col].mean().sort_values(ascending=False).head(15)

    sns.barplot(x=data.index, y=data.values, palette="viridis")
    plt.title(title, fontsize=14)
    plt.xlabel(geo_col)
    plt.ylabel(f"Average {value_col}")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_scatter_trend(
    df: pd.DataFrame, x_col: str, y_col: str, title: str, hue_col: str = None
):
    """
    Scatter plot with regression trendline. Supports `hue` for categories.
    """
    if x_col not in df.columns or y_col not in df.columns:
        return

    plt.figure(figsize=(10, 6))
    if len(df) > 10000:
        plot_data = df.sample(10000, random_state=42)
    else:
        plot_data = df

    if hue_col and hue_col in df.columns:
        sns.scatterplot(data=plot_data, x=x_col, y=y_col, hue=hue_col, alpha=0.4, s=15)
    else:
        sns.regplot(
            data=plot_data,
            x=x_col,
            y=y_col,
            scatter_kws={"alpha": 0.3, "s": 10},
            line_kws={"color": "red"},
        )

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_categorical_breakdown(
    df: pd.DataFrame, cat_col: str, value_col: str, title: str
):
    """
    Boxplot showing distribution across categories.
    """
    if cat_col not in df.columns or value_col not in df.columns:
        return

    plt.figure(figsize=(12, 6))
    top_cats = df[cat_col].value_counts().nlargest(10).index
    plot_data = df[df[cat_col].isin(top_cats)]

    sns.boxplot(data=plot_data, x=cat_col, y=value_col, palette="coolwarm")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis="y")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, cols: list):
    """
    Plots correlation heatmap for selected columns.
    """
    valid_cols = [c for c in cols if c in df.columns]
    if len(valid_cols) < 2:
        return

    corr = df[valid_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()


def plot_kde_distribution(df: pd.DataFrame, col: str):
    """
    Plots Kernel Density Estimation with histogram.
    """
    if col not in df.columns:
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True, color="purple", bins=30)
    plt.title(f"Distribution of {col} (Histogram + KDE)")
    plt.xlabel(col)
    plt.grid(axis="y", alpha=0.3)
    plt.show()
