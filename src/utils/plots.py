import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_numerical_distributions(
    df: pd.DataFrame, numerical_cols: list, save_path: str = None
):
    """
    Plots histograms for numerical columns.
    """
    if not numerical_cols:
        print("No numerical columns to plot.")
        return

    num_plots = len(numerical_cols)
    rows = (num_plots // 3) + 1
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame, numerical_cols: list, save_path: str = None
):
    """
    Plots heatmap of correlation matrix.
    """
    if not numerical_cols:
        return

    plt.figure(figsize=(12, 10))
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix")

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_outliers_boxplots(
    df: pd.DataFrame, numerical_cols: list, save_path: str = None
):
    """
    Plots boxplots to detect outliers.
    """
    if not numerical_cols:
        return

    if save_path:
        plt.savefig(save_path)
    plt.show()
