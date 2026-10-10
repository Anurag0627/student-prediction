"""Central configuration for the Student Outcome Prediction project.

Keeping constants in one place makes experiments reproducible and prevents
"magic numbers" from being scattered across the codebase. Import from here
rather than re-declaring paths or seeds in individual scripts/notebooks.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

# ---------------------------------------------------------------------------
# Dataset columns
# ---------------------------------------------------------------------------
# Discovered by inspecting the CSV (see notebook 01_data_understanding).
# The target has 3 classes: Dropout, Enrolled, Graduate (imbalanced).
TARGET_COLUMN: str = "final_outcome"

# Unique per-row identifier. It carries no predictive signal and must NOT be
# used as a feature (10,000 unique values for 10,000 rows).
ID_COLUMN: str = "student_id"

# The three outcome classes, ordered from minority to majority. Fixing the
# order keeps confusion matrices and per-class reports consistent everywhere.
CLASS_LABELS: list[str] = ["Dropout", "Enrolled", "Graduate"]

# ---------------------------------------------------------------------------
# Project paths (resolved relative to this file, so they work from anywhere)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
RAW_DATASET_PATH: Path = RAW_DATA_DIR / "student_outcome_research_dataset.csv"

MODELS_DIR: Path = PROJECT_ROOT / "models"

RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
EXPERIMENTS_DIR: Path = RESULTS_DIR / "experiments"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"


def ensure_directories() -> None:
    """Create the output directories if they do not already exist.

    Safe to call repeatedly; useful at the start of a script or notebook so
    that saving figures/metrics never fails on a fresh checkout.
    """
    for directory in (
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        EXPERIMENTS_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
