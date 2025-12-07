import pandas as pd
import numpy as np
from scipy import stats


def check_chi2_independence(df: pd.DataFrame, col1: str, col2: str):
    """
    Performs Chi-Squared Test for Independence between two categorical variables.
    Use this to test if Risk (e.g. High/Low Claims) is independent of Group (e.g. Province).

    Returns:
        p_value, contingency_table, interpretation
    """
    # Create contingency table
    contingency = pd.crosstab(df[col1], df[col2])

    # Run test
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)

    # Interpret
    if p_val < 0.05:
        interp = "Reject Null Hypothesis: Significant difference exists."
    else:
        interp = "Fail to Reject Null: No significant difference found."

    return p_val, contingency, interp


def check_ttest_means(
    df: pd.DataFrame, group_col: str, value_col: str, group_a: str, group_b: str
):
    """
    Performs Independent T-Test to compare means of 'value_col' between two groups.
    e.g. Compare 'Margin' between 'ZipCode A' and 'ZipCode B'.
    """
    sample_a = df[df[group_col] == group_a][value_col].dropna()
    sample_b = df[df[group_col] == group_b][value_col].dropna()

    if len(sample_a) < 2 or len(sample_b) < 2:
        return None, "Insufficient Data"

    stat, p_val = stats.ttest_ind(sample_a, sample_b, equal_var=False)  # Welch's t-test

    if p_val < 0.05:
        interp = "Reject Null Hypothesis: Means are significantly different."
    else:
        interp = "Fail to Reject Null: No significant difference in means."

    return p_val, interp


def check_anova(df: pd.DataFrame, group_col: str, value_col: str):
    """
    Performs One-Way ANOVA to compare means across >2 groups.
    e.g. Compare 'TotalClaims' across 'Provinces'.
    """
    groups = [group[value_col].dropna() for name, group in df.groupby(group_col)]

    # ANOVA requires at least 2 groups
    if len(groups) < 2:
        return None, "Insufficient Groups"

    stat, p_val = stats.f_oneway(*groups)

    if p_val < 0.05:
        interp = "Reject Null Hypothesis: At least one group mean is different."
    else:
        interp = "Fail to Reject Null: No significant difference across groups."

    return p_val, interp
