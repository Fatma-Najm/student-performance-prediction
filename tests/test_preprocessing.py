"""
test_preprocessing.py
=====================
Unit tests for preprocessing functions in src/preprocessing.py

Run with:
    pytest tests/test_preprocessing.py -v
"""

import pandas as pd
import numpy as np
import pytest
import sys
import os

# Allow imports from the src/ directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import encode_features, add_pass_fail, scale_features


# ── Sample test data ────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Create a small mock DataFrame that mirrors the UCI dataset structure."""
    return pd.DataFrame({
        "school": ["GP", "MS", "GP"],
        "sex": ["M", "F", "M"],
        "age": [17, 18, 16],
        "address": ["U", "R", "U"],
        "famsize": ["GT3", "LE3", "GT3"],
        "Pstatus": ["T", "A", "T"],
        "Medu": [4, 2, 3],
        "Fedu": [4, 1, 3],
        "Mjob": ["teacher", "other", "health"],
        "Fjob": ["services", "other", "teacher"],
        "reason": ["course", "home", "reputation"],
        "guardian": ["mother", "father", "mother"],
        "studytime": [2, 1, 3],
        "failures": [0, 1, 0],
        "schoolsup": ["yes", "no", "no"],
        "famsup": ["no", "yes", "yes"],
        "paid": ["no", "no", "yes"],
        "activities": ["yes", "no", "yes"],
        "nursery": ["yes", "yes", "no"],
        "higher": ["yes", "yes", "yes"],
        "internet": ["yes", "no", "yes"],
        "romantic": ["no", "no", "yes"],
        "absences": [4, 2, 0],
        "G1": [12, 9, 14],
        "G2": [13, 8, 15],
        "G3": [14, 7, 15],
    })


# ── Tests ──────────────────────────────────────────────────

def test_encode_binary_columns(sample_df):
    """Binary yes/no columns should be converted to 1/0."""
    encoded = encode_features(sample_df)
    assert encoded["schoolsup"].isin([0, 1]).all(), "schoolsup should be 0 or 1"
    assert encoded["internet"].isin([0, 1]).all(), "internet should be 0 or 1"


def test_encode_removes_original_categoricals(sample_df):
    """Original multi-category columns should be replaced by dummies."""
    encoded = encode_features(sample_df)
    assert "school" not in encoded.columns, "'school' column should be one-hot encoded"
    assert "Mjob" not in encoded.columns, "'Mjob' column should be one-hot encoded"


def test_add_pass_fail_column(sample_df):
    """Pass/Fail column should be added correctly based on threshold."""
    df = add_pass_fail(sample_df, grade_col="G3", threshold=10)
    assert "pass_fail" in df.columns, "'pass_fail' column should exist"
    # G3=14 → pass(1), G3=7 → fail(0), G3=15 → pass(1)
    assert list(df["pass_fail"]) == [1, 0, 1], "Pass/Fail values incorrect"


def test_pass_fail_is_binary(sample_df):
    """Pass/Fail column should only contain 0 and 1."""
    df = add_pass_fail(sample_df)
    assert df["pass_fail"].isin([0, 1]).all(), "pass_fail should only have values 0 or 1"


def test_scale_features_shape(sample_df):
    """Scaled arrays should retain the same shape as input."""
    X = sample_df[["studytime", "failures", "absences", "G1", "G2"]].values
    X_train, X_test = X[:2], X[2:]
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape


def test_scale_features_mean(sample_df):
    """Training data mean should be approximately 0 after standard scaling."""
    X = sample_df[["studytime", "failures", "absences"]].values
    X_train, X_test = X[:2], X[2:]
    X_train_scaled, _, _ = scale_features(X_train, X_test)
    mean = np.abs(X_train_scaled.mean(axis=0))
    assert (mean < 1e-10).all(), "Scaled training data should have mean ≈ 0"