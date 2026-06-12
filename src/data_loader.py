"""
data_loader.py
==============
Utility functions for loading and validating the Student Performance dataset.

Usage:
    from src.data_loader import load_data, get_basic_info
"""

import pandas as pd
import os


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the student performance CSV file into a pandas DataFrame.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            "Please download it from the UCI ML Repository and place it in data/raw/"
        )

    # The UCI dataset uses semicolons as delimiters
    df = pd.read_csv(filepath)
    print(f" Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def get_basic_info(df: pd.DataFrame) -> None:
    """
    Print a structured summary of the dataset.

    Parameters:
        df (pd.DataFrame): The loaded dataset.
    """
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Shape        : {df.shape}")
    print(f"Total cells  : {df.size}")
    print(f"Missing vals : {df.isnull().sum().sum()}")
    print(f"Duplicates   : {df.duplicated().sum()}")
    print("\nColumn Data Types:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    print(df.head())


def split_features_target(df: pd.DataFrame, target_col: str = "G3"):
    """
    Separate the dataset into features (X) and target (y).

    Parameters:
        df (pd.DataFrame): The full dataset.
        target_col (str): Name of the target column. Default is 'G3'.

    Returns:
        tuple: (X, y) where X is the feature DataFrame and y is the target Series.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f" Features: {X.shape[1]} columns | Target: '{target_col}'")
    return X, y