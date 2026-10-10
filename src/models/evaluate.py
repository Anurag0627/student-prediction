"""Evaluation utilities for imbalanced multi-class classification.

Why not accuracy alone? With ~59% ``Graduate``, a model that always predicts
``Graduate`` scores ~59% accuracy while never identifying a single at-risk
``Dropout`` student -- useless for the research goal. We therefore emphasise:

  * **Macro F1** -- averages F1 across classes with equal weight, so the small
    ``Dropout`` class counts as much as the large ``Graduate`` class.
  * **Per-class recall** -- of the students who actually dropped out, how many
    did we catch? This is the metric that matters most for early intervention.
  * **Per-class precision** -- of those we flagged, how many were correct?
  * **Confusion matrix** -- shows exactly which classes get confused.

Weighted F1 and accuracy are reported for completeness/comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.utils import config


def compute_metrics(y_true, y_pred) -> dict:
    """Return the headline scalar metrics for one set of predictions.

    Returns
    -------
    dict
        ``accuracy``, ``macro_precision``, ``macro_recall``, ``macro_f1``,
        ``weighted_f1``. All values are rounded to 4 decimals. ``zero_division=0``
        avoids warnings/NaNs when a class receives no predictions.
    """
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "macro_recall": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
    }


def per_class_metrics(
    y_true, y_pred, labels: list[str] | None = None
) -> pd.DataFrame:
    """Return per-class precision, recall, F1 and support as a DataFrame."""
    if labels is None:
        labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {
            "precision": precision.round(4),
            "recall": recall.round(4),
            "f1": f1.round(4),
            "support": support,
        },
        index=labels,
    )


def confusion(y_true, y_pred, labels: list[str] | None = None) -> pd.DataFrame:
    """Return the confusion matrix as a labelled DataFrame (rows=true, cols=pred)."""
    if labels is None:
        labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )


def text_report(y_true, y_pred, labels: list[str] | None = None) -> str:
    """Return scikit-learn's classification report as a formatted string."""
    if labels is None:
        labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))
    return classification_report(y_true, y_pred, labels=labels, zero_division=0)


def macro_roc_auc(y_true, y_proba, labels: list[str] | None = None) -> float | None:
    """Return the macro one-vs-rest ROC-AUC, or ``None`` if it cannot be computed.

    Parameters
    ----------
    y_proba:
        Predicted class probabilities of shape ``(n_samples, n_classes)`` as
        produced by ``pipeline.predict_proba``. Column order must match
        ``labels`` (i.e. the fitted estimator's ``classes_``).
    """
    if labels is None:
        labels = config.CLASS_LABELS
    try:
        y_true_bin = label_binarize(y_true, classes=labels)
        return round(
            float(
                roc_auc_score(
                    y_true_bin,
                    np.asarray(y_proba),
                    average="macro",
                    multi_class="ovr",
                )
            ),
            4,
        )
    except (ValueError, IndexError):
        return None
