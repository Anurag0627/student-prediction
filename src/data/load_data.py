"""Data loading and basic validation utilities.

Functions here are deliberately small and side-effect free (except for the
print helpers) so they can be reused from both scripts and notebooks.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import config


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw dataset from ``path`` (defaults to the configured raw CSV).

    Parameters
    ----------
    path:
        Optional path to a CSV file. When ``None`` the project's configured
        :data:`~src.utils.config.RAW_DATASET_PATH` is used.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    """
    csv_path = Path(path) if path is not None else config.RAW_DATASET_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. "
            "Place the CSV under data/raw/ (see README)."
        )
    return pd.read_csv(csv_path)


def get_shape(df: pd.DataFrame) -> tuple[int, int]:
    """Return the ``(n_rows, n_columns)`` shape of the dataset."""
    return df.shape


def inspect_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column summary: dtype, non-null count, and unique count.

    Handy for a quick structural overview without printing the whole frame.
    """
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "n_unique": df.nunique(),
        }
    )


def identify_target(df: pd.DataFrame, target_column: str = config.TARGET_COLUMN) -> str:
    """Confirm the target column exists and return its name.

    Raising early with a clear message prevents silent mistakes later in the
    pipeline (e.g. a renamed column breaking every downstream step).
    """
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    return target_column


def basic_validation(df: pd.DataFrame, target_column: str = config.TARGET_COLUMN) -> dict:
    """Run lightweight sanity checks and return a structured report.

    This does not raise on data-quality issues; it reports them so a notebook
    can display the findings and decide how to react.

    Returns
    -------
    dict
        Keys: ``n_rows``, ``n_columns``, ``total_missing``, ``n_duplicates``,
        ``target_present``, ``target_classes``, ``class_counts``.
    """
    target_present = target_column in df.columns
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "total_missing": int(df.isna().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "target_present": target_present,
        "target_classes": (
            sorted(df[target_column].dropna().unique().tolist())
            if target_present
            else []
        ),
        "class_counts": (
            df[target_column].value_counts().to_dict() if target_present else {}
        ),
    }
