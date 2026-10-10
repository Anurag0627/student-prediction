"""Preprocessing utilities: feature typing, train/test split, and pipelines.

Design goal: make data leakage *hard to introduce*.

  * :func:`split_data` splits BEFORE any scaling or resampling.
  * :func:`build_preprocessor` returns an *unfitted* transformer. It is meant
    to be fit inside a pipeline on the training fold only, so scaling and
    encoding parameters are never learned from the test set.

Never fit the preprocessor on the full dataset and then split.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import config


def separate_features_target(
    df: pd.DataFrame,
    target_column: str = config.TARGET_COLUMN,
    id_column: str | None = config.ID_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into features ``X`` and target ``y``.

    The identifier column (if present) is dropped from the features because it
    is unique per row and carries no predictive signal. The target column is
    never included in ``X``.
    """
    drop_cols = [target_column]
    if id_column is not None and id_column in df.columns:
        drop_cols.append(id_column)

    X = df.drop(columns=drop_cols)
    y = df[target_column]
    return X, y


def identify_numerical_features(X: pd.DataFrame) -> list[str]:
    """Return the names of numeric (int/float) feature columns."""
    return X.select_dtypes(include=["number"]).columns.tolist()


def identify_categorical_features(X: pd.DataFrame) -> list[str]:
    """Return the names of categorical (object/category/bool) feature columns."""
    return X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Stratification preserves the class proportions in both splits, which
    matters for an imbalanced target: without it the minority ``Dropout``
    class could be under-represented in the test set.

    Returns ``(X_train, X_test, y_train, y_test)``.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build an *unfitted* ``ColumnTransformer`` for the given feature groups.

    Numeric columns are standardized (optional) and categorical columns are
    one-hot encoded. ``handle_unknown="ignore"`` keeps the transformer robust
    to categories that appear only in the test set.

    Fit this inside a :class:`~sklearn.pipeline.Pipeline` so its parameters are
    learned from training data only.
    """
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def build_sklearn_pipeline(preprocessor: ColumnTransformer, estimator) -> Pipeline:
    """Wrap a preprocessor and an estimator into a single leakage-safe pipeline.

    Use this for the baseline and class-weighting experiments. For resampling
    experiments (oversampling / SMOTE) use imbalanced-learn's pipeline instead
    (see :mod:`src.models.train`) so resampling is applied to training folds
    only.
    """
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
