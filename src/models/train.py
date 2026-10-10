"""Training and experiment-execution utilities.

The unit of research here is an *experiment*::

    model  +  imbalance strategy  +  dataset split  +  evaluation  =  experiment

:func:`run_experiment` wires these together into a leakage-safe pipeline and
returns a structured result (metrics + metadata) ready to be stored as a row
in a results table.

Supported imbalance strategies
------------------------------
``"none"``               Baseline. No handling (Experiment A).
``"class_weight"``       Uses estimators built with ``class_weight="balanced"``
                         (Experiment B). Pass such estimators in directly.
``"random_oversample"``  ``RandomOverSampler`` on the training fold only
                         (Experiment C).
``"smote"``              ``SMOTE`` on the training fold only (Experiment D).

The strategy registry is intentionally small and easy to extend (Experiment E:
undersampling, SMOTE variants, ...).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data import preprocessing
from src.models import evaluate
from src.utils import config

# Resamplers that operate on the training fold only. "none" and "class_weight"
# do not resample, so they map to None.
_RESAMPLERS = {
    "none": None,
    "class_weight": None,
    "random_oversample": RandomOverSampler,
    "smote": SMOTE,
}


def available_strategies() -> list[str]:
    """Return the list of supported imbalance-strategy names."""
    return list(_RESAMPLERS)


def build_experiment_pipeline(
    preprocessor: ColumnTransformer,
    estimator,
    strategy: str = "none",
    random_state: int = config.RANDOM_STATE,
) -> ImbPipeline:
    """Build a leakage-safe pipeline for a given imbalance strategy.

    Resampling (when requested) is inserted *between* preprocessing and the
    model inside an imbalanced-learn pipeline. During cross-validation and at
    ``fit`` time, imbalanced-learn applies the resampler to the training data
    only -- never to the validation/test fold.
    """
    if strategy not in _RESAMPLERS:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Available: {available_strategies()}"
        )

    steps = [("preprocessor", preprocessor)]
    resampler_cls = _RESAMPLERS[strategy]
    if resampler_cls is not None:
        steps.append(("resampler", resampler_cls(random_state=random_state)))
    steps.append(("model", clone(estimator)))
    return ImbPipeline(steps=steps)


def train_model(pipeline, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit a pipeline/estimator on the training data and return it."""
    pipeline.fit(X_train, y_train)
    return pipeline


def predict(pipeline, X) -> pd.Series:
    """Return predictions for ``X`` from a fitted pipeline."""
    return pipeline.predict(X)


def cross_validate_pipeline(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "f1_macro",
    n_splits: int = 5,
    random_state: int = config.RANDOM_STATE,
) -> dict:
    """Run stratified k-fold cross-validation on the *training* data.

    Always pass training data here, never the held-out test set. Returns the
    per-fold scores plus their mean and standard deviation.
    """
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    scores = cross_val_score(pipeline, X, y, scoring=scoring, cv=cv)
    return {
        "scoring": scoring,
        "scores": scores.tolist(),
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
    }


def run_experiment(
    model_name: str,
    estimator,
    strategy: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    dataset_version: str = "raw",
    random_state: int = config.RANDOM_STATE,
) -> dict:
    """Run one full experiment and return a structured result dict.

    The dict includes the identifying metadata (model, strategy, dataset
    version, random seed, timestamp) alongside the evaluation metrics, so it
    can be appended directly to a results table.
    """
    numerical = preprocessing.identify_numerical_features(X_train)
    categorical = preprocessing.identify_categorical_features(X_train)
    preprocessor = preprocessing.build_preprocessor(numerical, categorical)

    pipeline = build_experiment_pipeline(
        preprocessor, estimator, strategy=strategy, random_state=random_state
    )
    pipeline = train_model(pipeline, X_train, y_train)
    y_pred = predict(pipeline, X_test)

    metrics = evaluate.compute_metrics(y_test, y_pred)

    result = {
        "model": model_name,
        "strategy": strategy,
        "dataset_version": dataset_version,
        "random_state": random_state,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **metrics,
    }
    return result


def results_to_frame(results: list[dict]) -> pd.DataFrame:
    """Convert a list of experiment-result dicts into a tidy DataFrame."""
    return pd.DataFrame(results)
