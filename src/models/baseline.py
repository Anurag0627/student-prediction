"""Baseline model configuration.

A single factory function returns fresh, untrained estimators. Nothing is
trained at import time. Two variants are provided so imbalance experiments can
reuse the same registry:

  * :func:`get_models` -- default estimators (Experiment A: baseline).
  * :func:`get_class_weighted_models` -- the same estimators configured with
    ``class_weight="balanced"`` where supported (Experiment B).

Adding a new model is a one-line change in the dictionaries below.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.utils import config


def get_models(random_state: int = config.RANDOM_STATE) -> dict:
    """Return a fresh dictionary of untrained baseline estimators.

    Called with no imbalance handling (Experiment A). Hyperparameters are left
    close to scikit-learn defaults on purpose: the goal at this stage is a
    reliable baseline, not a tuned model.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(random_state=random_state),
        "knn": KNeighborsClassifier(),
        # SVC keeps scikit-learn defaults. For ROC-AUC on SVM later, wrap it in
        # CalibratedClassifierCV(SVC(), ensemble=False) to obtain probabilities
        # (SVC(probability=True) is deprecated since scikit-learn 1.9).
        "svm": SVC(random_state=random_state),
    }


def get_class_weighted_models(random_state: int = config.RANDOM_STATE) -> dict:
    """Return baseline estimators with ``class_weight="balanced"`` where supported.

    Used for Experiment B (class weighting). KNN has no ``class_weight``
    parameter, so it is returned unchanged and should be interpreted as a
    baseline reference within this experiment.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced", random_state=random_state
        ),
        "knn": KNeighborsClassifier(),  # no class_weight support
        "svm": SVC(class_weight="balanced", random_state=random_state),
    }
