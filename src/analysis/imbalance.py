"""Class-imbalance analysis utilities.

These quantify *how* imbalanced the target is. Understanding the imbalance is
the starting point of this project's central research question, so keep these
metrics separate from general EDA.
"""

from __future__ import annotations

import pandas as pd

from src.utils import config


def class_counts(y: pd.Series) -> pd.Series:
    """Return the number of samples per class, sorted from largest to smallest."""
    return y.value_counts()


def class_percentages(y: pd.Series) -> pd.Series:
    """Return the percentage of samples per class (rounded to 2 decimals)."""
    return (y.value_counts(normalize=True) * 100).round(2)


def imbalance_ratio(y: pd.Series) -> float:
    """Return the imbalance ratio: majority class count / minority class count.

    A ratio of 1.0 means perfectly balanced. Larger values mean more severe
    imbalance. For this dataset the majority class is ``Graduate`` and the
    minority is ``Dropout``.
    """
    counts = y.value_counts()
    return round(float(counts.max() / counts.min()), 3)


def imbalance_report(
    y: pd.Series, target_column: str = config.TARGET_COLUMN
) -> pd.DataFrame:
    """Return a combined count + percentage table for the target classes."""
    counts = class_counts(y)
    percentages = class_percentages(y)
    return pd.DataFrame(
        {"count": counts, "percentage": percentages}
    ).rename_axis(target_column)


def compare_distributions(
    y_before: pd.Series, y_after: pd.Series
) -> pd.DataFrame:
    """Compare class counts before and after a resampling step.

    Useful for the imbalance-handling notebook to show, for example, how
    ``RandomOverSampler`` or ``SMOTE`` change the training-set distribution.
    Test-set distributions must never be resampled.
    """
    before = class_counts(y_before).rename("before")
    after = class_counts(y_after).rename("after")
    return pd.concat([before, after], axis=1).fillna(0).astype(int)
