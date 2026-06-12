"""
evaluate.py
===========
Functions for evaluating and comparing ML model performance.

Covers:
    - Classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC
    - Regression metrics: MAE, RMSE, R² Score
    - Comparison table for multiple models

Usage:
    from src.evaluate import evaluate_classifier, evaluate_regressor, compare_models
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)


def evaluate_classifier(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluate a classification model and print a performance summary.

    Parameters:
        model: Trained sklearn classifier.
        X_test: Test feature matrix.
        y_test: True test labels.
        model_name (str): Display name for the model.

    Returns:
        dict: Dictionary of metric names and values.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else "N/A",
    }

    print(f"\n{'='*45}")
    print(f"  Classification Report: {model_name}")
    print(f"{'='*45}")
    for key, val in metrics.items():
        if key != "Model":
            print(f"  {key:<15}: {val}")
    print(f"\n  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def evaluate_regressor(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluate a regression model and print a performance summary.

    Parameters:
        model: Trained sklearn regressor.
        X_test: Test feature matrix.
        y_test: True continuous target values.
        model_name (str): Display name for the model.

    Returns:
        dict: Dictionary of metric names and values.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "Model": model_name,
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "R² Score": round(r2_score(y_test, y_pred), 4),
    }

    print(f"\n{'='*45}")
    print(f"  Regression Report: {model_name}")
    print(f"{'='*45}")
    for key, val in metrics.items():
        if key != "Model":
            print(f"  {key:<15}: {val}")

    return metrics


def compare_models(results_list: list) -> pd.DataFrame:
    """
    Build a comparison table from a list of evaluation result dictionaries.

    Parameters:
        results_list (list): List of metric dicts returned by evaluate_classifier
                             or evaluate_regressor.

    Returns:
        pd.DataFrame: Formatted comparison table.
    """
    df = pd.DataFrame(results_list)
    df = df.set_index("Model")
    print("\n Model Comparison Table:")
    print(df.to_string())
    return df