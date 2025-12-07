import pandas as pd
import os


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from a CSV or pipe-delimited text file.
    """
    if not os.path.exists(filepath):
        # Fallback: try different relative paths
        if os.path.exists(os.path.join("..", filepath)):
            filepath = os.path.join("..", filepath)
        elif os.path.exists(os.path.join("data", "raw", "MachineLearningRating.txt")):
            filepath = os.path.join("data", "raw", "MachineLearningRating.txt")

    try:
        df = pd.read_csv(filepath, sep="|", low_memory=False)
    except Exception:
        df = pd.read_csv(filepath, sep=",", low_memory=False)

    return df
