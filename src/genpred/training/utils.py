"""Utils for training."""

from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

# pylint:disable=invalid-name

METRICS_MAPPER = {
    "mean_train_accuracy": "train_accuracy",
    "mean_test_accuracy": "val_accuracy",
    "mean_train_matthews_corrcoef": "train_matthews_corrcoef",
    "mean_test_matthews_corrcoef": "val_matthews_corrcoef",
    "mean_train_roc_auc": "train_roc_auc",
    "mean_test_roc_auc": "val_roc_auc",
}


RANK_COLS = [
    "rank_test_accuracy",
    "rank_test_matthews_corrcoef",
    "rank_test_roc_auc",
]

METRICS = [
    "train_accuracy",
    "val_accuracy",
    "test_accuracy",
    "train_matthews_corrcoef",
    "val_matthews_corrcoef",
    "test_matthews_corrcoef",
    "train_roc_auc",
    "val_roc_auc",
    "test_roc_auc",
]


def refit_strategy(cv_results: Dict) -> int:
    """Implements refit strategy."""
    results = pd.DataFrame(cv_results)
    best_rank = results["rank_test_accuracy"] == 1
    return int(results[best_rank].index[0])


def save_model(model: RandomizedSearchCV) -> None:
    """Saves model in pickle format."""
    joblib.dump(model, "model.pkl", protocol=5)


def save_cv_results(cv_results: Dict) -> None:
    """Saves CV results in csv format."""
    results = pd.DataFrame(cv_results)
    results = results[[c for c in results.columns if "std" not in c]]
    results = results[[c for c in results.columns if "split" not in c]]
    results = results.drop(["params"], axis=1)
    results.round(3).to_csv("cv_results.csv", index=False)

    metrics = results.iloc[[refit_strategy(cv_results)], :]
    metrics = metrics.rename(columns=METRICS_MAPPER)
    metrics[list(METRICS_MAPPER.values())].round(3).to_csv("metrics.csv", index=False)


def compute_predictions(
    estimator: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    genomes_test: List,
) -> pd.DataFrame:
    """Computes predictions."""
    probas = estimator.predict_proba(X_test)[:, 1]
    predictions = pd.DataFrame(
        {
            "Genome": genomes_test,
            "Target": y_test,
            "Prediction": probas.round().astype(int),
            "Probability": probas,
        }
    )
    predictions.to_csv("predictions.csv", index=False)
    return predictions


def save_test_metrics(
    estimator: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    genomes_test: List,
) -> None:
    """Computes metrics on a test set."""
    predictions = compute_predictions(estimator, X_test, y_test, genomes_test)
    preds = predictions.Prediction.values
    probas = predictions.Probability.values
    targets = predictions.Target.values

    metrics = pd.DataFrame(
        {
            "test_accuracy": [accuracy_score(targets, preds)],
            "test_f1_weighted": [f1_score(targets, preds, average="weighted")],
            "test_matthews_corrcoef": [matthews_corrcoef(targets, preds)],
            "test_roc_auc": [roc_auc_score(targets, probas)],
        }
    )

    cv_metrics = pd.read_csv("metrics.csv")
    metrics = pd.concat([cv_metrics, metrics.round(3)], axis=1)
    metrics[METRICS].to_csv("metrics.csv", index=False)


def check_attr(estimator: Pipeline, attr: str) -> None:
    """Checks tuner is fitted and has the right attributes."""
    assert hasattr(estimator, "named_steps")
    assert attr in estimator.named_steps


def extract_vocabulary(estimator: Pipeline) -> Dict:
    """Extracts vocabulary from tuner."""
    check_attr(estimator, attr="vectorizer")
    vectorizer = estimator.named_steps.get("vectorizer")
    return vectorizer.vocabulary_


def extract_feature_importances(estimator: Pipeline) -> np.ndarray:
    """Extracts feature importances from tuner"""
    check_attr(estimator, attr="classifier")
    classifier = estimator.named_steps.get("classifier")
    return classifier.feature_importances_


def save_feature_importances(estimator: Pipeline) -> None:
    """Gets feature importances from tuner and saves them as a DataFrame."""
    vocabulary = extract_vocabulary(estimator)
    importances = extract_feature_importances(estimator)
    fis = pd.DataFrame({"Word": vocabulary.keys(), "Importance": importances})
    fis = fis.sort_values("Importance", ascending=False)
    fis.to_csv("importances.csv", index=False)
