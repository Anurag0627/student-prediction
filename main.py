"""Project entry point / smoke test.

Runs the *baseline* experiment (Experiment A: no imbalance handling) for every
registered model on a proper stratified train/test split, prints the headline
metrics, and saves them to ``results/metrics/baseline_results.csv``.

This is intentionally minimal -- it demonstrates that the foundation works
end-to-end. The richer, story-driven analysis lives in the notebooks.

Run with::

    python main.py
"""

from __future__ import annotations

from src.analysis import imbalance
from src.data import load_data, preprocessing
from src.models import baseline, evaluate, train
from src.utils import config


def main() -> None:
    config.ensure_directories()

    # 1. Load and validate ---------------------------------------------------
    df = load_data.load_dataset()
    report = load_data.basic_validation(df)
    print("=== Dataset validation ===")
    for key, value in report.items():
        print(f"  {key}: {value}")

    # 2. Features / target + imbalance snapshot ------------------------------
    X, y = preprocessing.separate_features_target(df)
    print("\n=== Class imbalance (full dataset) ===")
    print(imbalance.imbalance_report(y))
    print(f"  imbalance ratio (majority/minority): {imbalance.imbalance_ratio(y)}")

    # 3. Split BEFORE any resampling ----------------------------------------
    X_train, X_test, y_train, y_test = preprocessing.split_data(X, y)
    print(f"\nTrain shape: {X_train.shape} | Test shape: {X_test.shape}")

    # 4. Baseline experiment for every model --------------------------------
    print("\n=== Baseline experiments (strategy='none') ===")
    results = []
    for name, estimator in baseline.get_models().items():
        result = train.run_experiment(
            model_name=name,
            estimator=estimator,
            strategy="none",
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            dataset_version="raw",
        )
        results.append(result)
        print(
            f"  {name:20s} "
            f"acc={result['accuracy']:.3f}  "
            f"macro_f1={result['macro_f1']:.3f}  "
            f"macro_recall={result['macro_recall']:.3f}"
        )

    # 5. Persist results -----------------------------------------------------
    results_df = train.results_to_frame(results)
    out_path = config.METRICS_DIR / "baseline_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved baseline metrics -> {out_path}")

    # 6. Per-class detail for the best baseline (by macro F1) ----------------
    best_row = results_df.sort_values("macro_f1", ascending=False).iloc[0]
    print(f"\nBest baseline by macro F1: {best_row['model']}")
    print("(Per-class detail is explored further in the notebooks.)")


if __name__ == "__main__":
    main()
