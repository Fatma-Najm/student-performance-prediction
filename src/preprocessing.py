"""
preprocessing.py
================
Functions for cleaning and transforming the Student Performance dataset.

Steps covered:
    1. Encode binary categorical columns (yes/no → 1/0)
    2. One-hot encode multi-category columns
    3. Add a binary classification target (Pass/Fail)
    4. Scale numerical features

Usage:
    from src.preprocessing import encode_features, add_pass_fail, scale_features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Columns with binary yes/no values — convert to 1/0
BINARY_COLUMNS = [
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]

# Columns with multiple categories — apply one-hot encoding
CATEGORICAL_COLUMNS = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian"]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns into numerical format.

    - Binary columns (yes/no) → mapped to (1/0)
    - Multi-category columns → one-hot encoded (drop_first=True to avoid multicollinearity)

    Parameters:
        df (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Encoded dataset.
    """
    df = df.copy()

    # Step 1: Map binary yes/no columns to 1/0
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0})

    # Step 2: One-hot encode multi-category columns
    existing_cats = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
    df = pd.get_dummies(df, columns=existing_cats, drop_first=True)

    print(f" Encoding complete. Dataset now has {df.shape[1]} columns.")
    return df


def add_pass_fail(df: pd.DataFrame, grade_col: str = "G3", threshold: int = 10) -> pd.DataFrame:
    """
    Add a binary 'pass_fail' column based on the final grade.

    Parameters:
        df (pd.DataFrame): Dataset with grade column.
        grade_col (str): Name of the grade column. Default is 'G3'.
        threshold (int): Minimum grade to pass. Default is 10.

    Returns:
        pd.DataFrame: Dataset with new 'pass_fail' column (1=Pass, 0=Fail).
    """
    df = df.copy()
    df["pass_fail"] = (df[grade_col] >= threshold).astype(int)

    pass_rate = df["pass_fail"].mean() * 100
    print(f" Pass/Fail column added. Pass rate: {pass_rate:.1f}%")
    return df


def scale_features(X_train, X_test):
    """
    Standardize numerical features using StandardScaler.

    Fits the scaler ONLY on training data, then transforms both train and test.
    This prevents data leakage from the test set.

    Parameters:
        X_train: Training feature matrix.
        X_test: Testing feature matrix.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    scaler = StandardScaler()

    # Fit only on training data — never on test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(" Features scaled using StandardScaler.")
    return X_train_scaled, X_test_scaled, scaler