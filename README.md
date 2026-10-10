# Enhancing Student Outcome Prediction

**The Role of Class Imbalance Handling in Machine Learning Models**

## Project overview

This is a research-oriented machine learning project that predicts a student's
final academic outcome — **Dropout**, **Enrolled**, or **Graduate** — from
academic, engagement, and support-service features. The dataset is imbalanced
(the `Graduate` class dominates), which is not just a nuisance to work around
but the **central object of study** here.

## Research problem

In real institutions, the students who most need early intervention — those at
risk of **dropping out** — are the *minority* class. A model optimised for raw
accuracy can look excellent while completely failing to identify at-risk
students. This project investigates how to build models that are reliable for
*all* outcome classes, not just the majority.

## Research question

> **How does class imbalance handling affect the performance and reliability of
> machine learning models for predicting Dropout, Enrolled, and Graduate
> student outcomes?**

## Objectives

1. Establish a reliable baseline on the original imbalanced data.
2. Quantify the imbalance and its effect on per-class performance.
3. Compare imbalance-handling techniques (class weighting, random
   oversampling, SMOTE) under a leakage-safe protocol.
4. Evaluate with metrics appropriate for imbalanced multi-class problems.
5. Provide an extensible foundation for future techniques and analysis.

## Dataset description

- **File:** `data/raw/student_outcome_research_dataset.csv`
- **Rows:** 10,000 &nbsp;|&nbsp; **Columns:** 53
- **Quality:** no missing values, no duplicate rows, no invalid values
- **Identifier:** `student_id` (unique per row — dropped, never a feature)
- **Target:** `final_outcome` with three classes

| Class    | Count | Percentage |
|----------|------:|-----------:|
| Graduate |  5,928 | 59.28% |
| Enrolled |  2,697 | 26.97% |
| Dropout  |  1,375 | 13.75% |

Imbalance ratio (majority / minority) ≈ **4.31**.

Feature groups include prior academic history (GPA, credits, backlogs),
current-term assessments (internal, midterm, practical), attendance,
LMS/engagement activity (Moodle logins, video/material views, forum activity),
and support-service participation (mentoring, counselling, remedial classes).

## Machine learning approach

- Supervised **multi-class classification** (3 classes).
- Stratified train/test split **before** any resampling.
- scikit-learn / imbalanced-learn **pipelines** so preprocessing and
  resampling are learned from training data only.
- Baseline first, then controlled comparison of imbalance-handling techniques.

Initial model roster:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors
5. Support Vector Machine

## The class imbalance problem

Because `Graduate` makes up ~59% of the data, a trivial model that always
predicts `Graduate` reaches ~59% accuracy while achieving **zero recall** on
`Dropout`. Accuracy alone is therefore misleading. Imbalance handling aims to
improve minority-class recall/F1 — but it can trade off precision or majority
performance. Measuring those trade-offs honestly is the point of this project.

## Planned experiments

| ID | Experiment          | Technique                                   | Status  |
|----|---------------------|---------------------------------------------|---------|
| A  | Baseline            | None (original imbalanced data)             | Foundation ready |
| B  | Class weighting     | `class_weight="balanced"`                   | Foundation ready |
| C  | Random oversampling | `RandomOverSampler` (train fold only)       | Foundation ready |
| D  | SMOTE               | `SMOTE` (train fold only)                   | Foundation ready |
| E  | Extensions          | Undersampling, SMOTE variants, …            | Extensible |

Each experiment is defined as: **model × imbalance strategy × dataset split ×
evaluation**. Results carry metadata (model, strategy, dataset version, random
seed, timestamp).

> **No experimental conclusions are claimed yet.** The techniques above are
> *implemented and comparable*; which one helps, and for which class, is to be
> determined by actually running the experiments.

## Evaluation metrics

Reported for every experiment: **accuracy, macro precision, macro recall,
macro F1, weighted F1**, plus **per-class precision/recall/F1**, the
**confusion matrix**, and the **classification report**. ROC-AUC (one-vs-rest,
macro) is supported for models exposing probabilities.

**Primary metrics for comparing imbalance techniques:** macro F1, per-class
recall, per-class precision, and the confusion matrix — because they weight the
minority `Dropout` class equally and reveal *which* classes are confused,
rather than hiding failures behind majority-driven accuracy.

## Project structure

```
student-outcome-prediction/
├── data/
│   ├── raw/        # original dataset (kept locally)
│   └── processed/  # generated intermediate data (git-ignored)
├── notebooks/      # 01..08 — the research story, in order
├── src/
│   ├── data/       # load_data.py, preprocessing.py
│   ├── analysis/   # eda.py, imbalance.py
│   ├── models/     # baseline.py, train.py, evaluate.py
│   ├── visualization/  # plots.py
│   └── utils/      # config.py (seeds, paths, target column)
├── models/         # saved model binaries (git-ignored)
├── results/        # figures/, metrics/, experiments/
├── reports/        # written analysis / research reporting
├── main.py         # end-to-end baseline smoke test
├── requirements.txt
└── README.md
```

## Installation

```bash
# From the project root. A virtual environment is recommended.
python -m venv .venv

# Activate it:
#   Windows (bash):      source .venv/Scripts/activate
#   Windows (PowerShell): .venv\Scripts\Activate.ps1
#   macOS / Linux:        source .venv/bin/activate

pip install -r requirements.txt
```

## How to run the Python scripts

```bash
# End-to-end baseline smoke test (loads data, splits, trains all 5 models,
# saves results/metrics/baseline_results.csv):
python main.py
```

Modules under `src/` are importable, e.g.:

```python
from src.data import load_data, preprocessing
from src.models import baseline, train, evaluate
```

## How to run the notebooks

```bash
jupyter notebook   # or: jupyter lab
```

Work through them in order — they tell the research story:

1. `01_data_understanding` — shape, columns, types, target, quality checks
2. `02_eda` — distributions and relationships
3. `03_preprocessing` — build the leakage-safe pipeline
4. `04_baseline_models` — Experiment A (no imbalance handling)
5. `05_class_imbalance_analysis` — why accuracy misleads here
6. `06_imbalance_handling` — Experiments B/C/D compared
7. `07_model_comparison` — model × technique × metrics
8. `08_final_analysis` — interpretation, trade-offs, limitations, future work

## Reproducibility

A single seed (`RANDOM_STATE = 42`) and split (`TEST_SIZE = 0.2`) live in
`src/utils/config.py`, along with the discovered target column
(`final_outcome`). Import these constants everywhere instead of hardcoding.

## Dataset handling

The raw CSV under `data/raw/` is **not** git-ignored: this is a self-contained
research project that needs the data locally to reproduce every result, and the
file is small enough to keep. Generated artifacts *are* ignored — processed
data, trained model binaries, and figures can all be regenerated from the code.
Small metrics/experiment tables under `results/` are kept by default because
they are the record of the research. If this dataset were sensitive or license
-restricted, the appropriate change would be to ignore `data/raw/` and document
how to obtain the file — adjust `.gitignore` accordingly if that becomes true.

## Future work

- Add Experiment E techniques (random undersampling, SMOTE variants such as
  BorderlineSMOTE/ADASYN, and combined over/under-sampling).
- Threshold tuning and cost-sensitive learning for the minority class.
- Cross-validated hyperparameter search once baselines are understood.
- Statistical significance testing across techniques and seeds.
- Calibration analysis and richer error analysis per class.
- Written research report under `reports/`.

## Data leakage safeguards (by design)

- Train/test split happens **before** any resampling or scaling.
- Resampling (oversampling/SMOTE) is applied to **training folds only**, via
  imbalanced-learn pipelines — never to the test set.
- Preprocessing parameters (scaling, encoding) are fit inside pipelines on
  training data only.
- The target is never used as a feature; the unique `student_id` is dropped.

