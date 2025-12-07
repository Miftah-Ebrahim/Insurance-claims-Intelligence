
import pandas as pd
import os

def load_data(filepath: str, sep: str = '|') -> pd.DataFrame:
    """
    Loads the insurance dataset from a text file.
    
    Args:
        filepath (str): Path to the dataset file.
        sep (str): Delimiter used in the file. Defaults to '|' based on common formats, 
                   but adaptable if checks reveal otherwise.
                   
    Returns:
        pd.DataFrame: Loaded dataframe.
        
    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    try:
        # read_csv with low_memory=False to handle mixed types warning initially
        df = pd.read_csv(filepath, sep=sep, low_memory=False)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
