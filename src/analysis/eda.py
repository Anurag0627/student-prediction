"""Exploratory Data Analysis (EDA) helper functions.

These return plain data structures (Series / DataFrames) rather than drawing
plots, so they stay reusable and testable. Plotting lives in
:mod:`src.visualization.plots`.
"""

from __future__ import annotations

import pandas as pd

from src.utils import config


def target_distribution(
    df: pd.DataFrame, target_column: str = config.TARGET_COLUMN
) -> pd.DataFrame:
    """Return counts and percentages for each target class."""
    counts = df[target_column].value_counts()
    percentages = (counts / counts.sum() * 100).round(2)
    return pd.DataFrame({"count": counts, "percentage": percentages})


def numerical_summary(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns (``df.describe``)."""
    numeric_df = df[columns] if columns is not None else df.select_dtypes("number")
    return numeric_df.describe().T


def categorical_summary(
    df: pd.DataFrame, columns: list[str] | None = None
) -> dict[str, pd.Series]:
    """Return value counts for each categorical column.

    Returns a mapping ``{column_name: value_counts_series}``.
    """
    if columns is None:
        columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return {col: df[col].value_counts() for col in columns}


def correlation_matrix(
    df: pd.DataFrame, columns: list[str] | None = None, method: str = "pearson"
) -> pd.DataFrame:
    """Return the correlation matrix of numeric columns."""
    numeric_df = df[columns] if columns is not None else df.select_dtypes("number")
    return numeric_df.corr(method=method)


def top_correlations(
    df: pd.DataFrame, threshold: float = 0.7, method: str = "pearson"
) -> pd.DataFrame:
    """Return unique numeric feature pairs whose |correlation| >= ``threshold``.

    Useful for spotting redundant features (e.g. attendance counts that are
    near-duplicates of an attendance percentage).
    """
    corr = correlation_matrix(df, method=method).abs()
    records = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if value >= threshold:
                records.append((cols[i], cols[j], round(float(value), 3)))
    result = pd.DataFrame(records, columns=["feature_a", "feature_b", "abs_correlation"])
    return result.sort_values("abs_correlation", ascending=False).reset_index(drop=True)
