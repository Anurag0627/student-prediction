"""Reusable plotting functions.

Each function draws onto a Matplotlib figure and, if ``save_path`` is given,
saves it under ``results/figures/`` (created on demand). Functions return the
Matplotlib ``Axes`` so notebooks can further customise the plot.

Seaborn is used for styling but the API stays thin -- no hidden global state.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import config


def _save(fig, save_path: str | Path | None) -> None:
    """Save ``fig`` to ``save_path`` (relative paths resolve to results/figures)."""
    if save_path is None:
        return
    path = Path(save_path)
    if not path.is_absolute():
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = config.FIGURES_DIR / path
    fig.savefig(path, bbox_inches="tight", dpi=150)


def plot_class_distribution(
    y: pd.Series,
    title: str = "Class distribution",
    save_path: str | Path | None = None,
):
    """Bar plot of the number of samples per class."""
    counts = y.value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("class")
    ax.set_ylabel("count")
    for i, value in enumerate(counts.values):
        ax.text(i, value, str(value), ha="center", va="bottom")
    _save(fig, save_path)
    return ax


def plot_confusion_matrix(
    confusion_df: pd.DataFrame,
    title: str = "Confusion matrix",
    save_path: str | Path | None = None,
):
    """Heatmap of a confusion matrix produced by ``evaluate.confusion``."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("true label")
    ax.set_xlabel("predicted label")
    _save(fig, save_path)
    return ax


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "macro_f1",
    group_by: str = "strategy",
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """Grouped bar chart comparing a metric across models and strategies.

    Expects a tidy results frame with at least ``model``, ``group_by`` and the
    ``metric`` columns (as produced by ``train.results_to_frame``).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=results_df, x="model", y=metric, hue=group_by, ax=ax)
    ax.set_title(title or f"{metric} by model and {group_by}")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    _save(fig, save_path)
    return ax


def plot_per_class_metric(
    per_class_df: pd.DataFrame,
    metric: str = "recall",
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """Bar plot of a per-class metric (e.g. recall or f1).

    ``per_class_df`` is the DataFrame returned by ``evaluate.per_class_metrics``
    (index = class labels).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=per_class_df.index, y=per_class_df[metric], ax=ax)
    ax.set_title(title or f"Per-class {metric}")
    ax.set_xlabel("class")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    _save(fig, save_path)
    return ax


def plot_resampling_comparison(
    comparison_df: pd.DataFrame,
    title: str = "Class distribution before vs after resampling",
    save_path: str | Path | None = None,
):
    """Grouped bar chart of class counts before/after resampling.

    ``comparison_df`` is the DataFrame returned by
    ``imbalance.compare_distributions`` (columns ``before`` and ``after``).
    """
    ax = comparison_df.plot(kind="bar", figsize=(7, 4))
    ax.set_title(title)
    ax.set_xlabel("class")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=0)
    _save(ax.get_figure(), save_path)
    return ax
