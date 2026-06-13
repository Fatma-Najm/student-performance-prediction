"""
train.py
========
Models included:
    - Logistic Regression (classification baseline)
    - Random Forest (ensemble — classification & regression)
    - Gradient Boosting (ensemble — classification & regression)

Usage:
    from src.train import train_classifier, train_regressor, save_model
"""

import joblib
import os
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


def train_classifier(X_train, y_train, model_name: str = "random_forest"):
    """
    Train a classification model.

    Parameters:
        X_train: Training feature matrix.
        y_train: Training labels (0=Fail, 1=Pass).
        model_name (str): One of 'logistic_regression', 'random_forest', 'gradient_boosting'.

    Returns:
        Trained sklearn model.
    """
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: '{model_name}'. Choose from {list(models.keys())}")

    model = models[model_name]
    model.fit(X_train, y_train)
    print(f" Classifier trained: {model_name}")
    return model


def train_regressor(X_train, y_train, model_name: str = "random_forest"):
    """
    Train a regression model to predict the exact final grade (G3).

    Parameters:
        X_train: Training feature matrix.
        y_train: Continuous target values (grades 0–20).
        model_name (str): One of 'random_forest', 'gradient_boosting'.

    Returns:
        Trained sklearn model.
    """
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: '{model_name}'. Choose from {list(models.keys())}")

    model = models[model_name]
    model.fit(X_train, y_train)
    print(f" Regressor trained: {model_name}")
    return model


def save_model(model, filename: str, save_dir: str = "models/") -> None:
    """
    Save a trained model to disk using joblib.

    Parameters:
        model: Trained sklearn model to save.
        filename (str): Filename for the saved model (e.g., 'random_forest.pkl').
        save_dir (str): Directory to save the model in. Default is 'models/'.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    joblib.dump(model, filepath)
    print(f" Model saved to: {filepath}")


def load_model(filename: str, save_dir: str = "models/"):
    """
    Load a previously saved model from disk.

    Parameters:
        filename (str): Filename of the saved model.
        save_dir (str): Directory where the model is stored.

    Returns:
        Loaded sklearn model.
    """
    filepath = os.path.join(save_dir, filename)
    model = joblib.load(filepath)
    print(f" Model loaded from: {filepath}")
    return model
