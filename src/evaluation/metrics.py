"""
Evaluation metrics for regression and classification tasks.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, median_absolute_error,
    accuracy_score, balanced_accuracy_score, f1_score,
    cohen_kappa_score, classification_report, confusion_matrix
)


def regression_metrics(y_true, y_pred) -> dict:
    """Compute standard regression metrics."""
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "MedAE": median_absolute_error(y_true, y_pred),
    }


def classification_metrics(y_true, y_pred) -> dict:
    """Compute standard classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, average="macro"),
        "Weighted_F1": f1_score(y_true, y_pred, average="weighted"),
        "Cohen_Kappa": cohen_kappa_score(y_true, y_pred),
    }


def print_regression_comparison(results: dict):
    """Pretty-print regression model comparison."""
    df = pd.DataFrame(results).T
    df = df.sort_values("R2", ascending=False)
    print(df.round(4).to_string())
    return df


def print_classification_comparison(results: dict):
    """Pretty-print classification model comparison."""
    df = pd.DataFrame(results).T
    df = df.sort_values("Weighted_F1", ascending=False)
    print(df.round(4).to_string())
    return df
