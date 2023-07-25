"""Metric functions."""
from typing import Callable, Dict

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    make_scorer,
)


def _compute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fun: Callable,
    round_values: bool = True,
    **kwargs,
) -> float:
    """General function to compute metric."""
    if round_values:
        y_pred = y_pred.round().astype(int)

    return fun(y_true, y_pred, **kwargs)


def acc_score(y_true: np.ndarray, y_pred: np.ndarray,) -> float:
    """Computes accuracy with optional aggregation."""
    return _compute(y_true, y_pred, accuracy_score, )


def mcc_score(y_true: np.ndarray, y_pred: np.ndarray,) -> float:
    """Computes weighted F1 with optional aggregation."""
    return _compute(y_true, y_pred, matthews_corrcoef, )


def auroc_score(y_true: np.ndarray, y_pred: np.ndarray,) -> float:
    """Computes ROC AUC with optional aggregation."""
    return _compute(y_true, y_pred, roc_auc_score, round_values=False)


def get_metrics_genome() -> Dict:
    """Returns metrics dictionary for genome strategy."""
    return {
        "accuracy": make_scorer(acc_score),
        "matthews_corrcoef": make_scorer(mcc_score),
        "roc_auc": make_scorer(auroc_score, needs_proba=True),
    }
