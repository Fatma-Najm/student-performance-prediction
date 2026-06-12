"""
visualize.py
============
Reusable plotting functions for EDA and model evaluation.

All functions optionally save figures to disk.

Usage:
    from src.visualize import plot_grade_distribution, plot_correlation_heatmap
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Apply a clean, professional plot style globally
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURES_DIR = "reports/figures/"


def _save_figure(filename: str) -> None:
    """Helper to save the current figure to the reports/figures directory."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, bbox_inches="tight", dpi=150)
    print(f" Figure saved: {filepath}")


def plot_grade_distribution(df: pd.DataFrame, grade_col: str = "G3", save: bool = True) -> None:
    """Plot the distribution of final grades."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Final Grade (G3) Distribution", fontsize=16, fontweight="bold")

    # Histogram
    axes[0].hist(df[grade_col], bins=20, color="#4C72B0", edgecolor="white", linewidth=0.8)
    axes[0].set_title("Grade Frequency")
    axes[0].set_xlabel("Final Grade (G3)")
    axes[0].set_ylabel("Count")

    # Pass/Fail pie chart
    if "pass_fail" in df.columns:
        counts = df["pass_fail"].value_counts()
        axes[1].pie(counts, labels=["Pass", "Fail"], autopct="%1.1f%%",
                    colors=["#4C72B0", "#DD8452"], startangle=90)
        axes[1].set_title("Pass vs Fail")

    plt.tight_layout()
    if save:
        _save_figure("grade_distribution.png")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> None:
    """Plot a heatmap of feature correlations (numerical columns only)."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Show only lower triangle
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save:
        _save_figure("correlation_heatmap.png")
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 15, save: bool = True) -> None:
    """
    Plot feature importances from a tree-based model (e.g., Random Forest).

    Parameters:
        model: Trained tree-based sklearn model with feature_importances_ attribute.
        feature_names (list): List of feature column names.
        top_n (int): Number of top features to display.
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.nlargest(top_n).sort_values()

    plt.figure(figsize=(10, 7))
    top_features.plot(kind="barh", color="#4C72B0", edgecolor="white")
    plt.title(f"Top {top_n} Feature Importances (Random Forest)", fontsize=14, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    if save:
        _save_figure("feature_importance.png")
    plt.show()


def plot_model_comparison(comparison_df: pd.DataFrame, metric: str = "Accuracy", save: bool = True) -> None:
    """
    Bar chart comparing multiple models on a given metric.

    Parameters:
        comparison_df (pd.DataFrame): Output from evaluate.compare_models().
        metric (str): Column name to plot (e.g., 'Accuracy', 'F1-Score').
    """
    plt.figure(figsize=(9, 5))
    comparison_df[metric].sort_values().plot(kind="barh", color="#4C72B0", edgecolor="white")
    plt.title(f"Model Comparison — {metric}", fontsize=14, fontweight="bold")
    plt.xlabel(metric)
    plt.xlim(0, 1.1)
    for i, val in enumerate(comparison_df[metric].sort_values()):
        plt.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=11)
    plt.tight_layout()
    if save:
        _save_figure("model_comparison.png")
    plt.show()