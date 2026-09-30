"""
Phase 3 - UP/DOWN classification benchmark.

Runs the CENTRAL evaluation harness (training/evaluation_harness.py) on
the classification track only - no separate evaluation logic lives here.

Target   : training/targets.py (direction = 1 if future_return > 0 else 0)
Features : technical_v2 (features/feature_engineering.py)
Validation: TimeSeriesSplit(n_splits=20, gap=horizon), pooled out-of-sample
Baselines: Always UP, Base Rate (the harness's classification baselines)
Models   : REQ-ML-001 - Logistic Regression, Random Forest, XGBoost, LightGBM
Gate     : unchanged gate_v1 (log loss < Base Rate with one-sided
           Diebold-Mariano p < 0.05, and accuracy >= Always UP)

Hyperparameters are FIXED in advance, not tuned. Tuning on these folds
would leak the evaluation into model selection; a tuned model would need
nested time-series validation (later work). The tree settings are
deliberately shallow/regularized because the audit showed unregularized
trees memorising noise (train R2 ~0.85-0.93 vs negative test R2).

Usage (from the project root):

    python -m training.train_classification
    python -m training.train_classification --horizon 5 --no-save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.evaluation_harness import (  # noqa: E402
    CLASSIFICATION,
    DEFAULT_DATASET,
    DEFAULT_RESULTS_DIR,
    Candidate,
    HarnessConfig,
    default_candidates,
    print_report,
    run_evaluation,
    save_results,
)

EXPERIMENT = "classification_technical"
SEED = 42


def classification_candidates() -> list[Candidate]:
    """Harness classification baselines + the REQ-ML-001 models."""
    baselines = [c for c in default_candidates() if c.task == CLASSIFICATION and c.is_baseline]
    models = [
        Candidate(
            "Logistic Regression", CLASSIFICATION,
            lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=1000)),
            ]),
            description="C=1.0, standardized features (same as Phase 1)",
        ),
        Candidate(
            "Random Forest", CLASSIFICATION,
            lambda: RandomForestClassifier(
                n_estimators=300, max_depth=3, min_samples_leaf=50,
                max_features="sqrt", random_state=SEED, n_jobs=1,
            ),
            description="300 trees, depth 3, min_samples_leaf 50 (fixed, regularized)",
        ),
        Candidate(
            "XGBoost", CLASSIFICATION,
            lambda: XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", random_state=SEED, n_jobs=1,
            ),
            description="300 rounds, depth 3, lr 0.03, subsample/colsample 0.8 (fixed)",
        ),
        Candidate(
            "LightGBM", CLASSIFICATION,
            lambda: LGBMClassifier(
                n_estimators=300, max_depth=3, num_leaves=7, learning_rate=0.03,
                min_child_samples=50, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, random_state=SEED, n_jobs=1,
                deterministic=True, verbose=-1,
            ),
            description="300 rounds, depth 3 / 7 leaves, lr 0.03, min_child_samples 50 (fixed)",
        ),
    ]
    return baselines + models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FindMarketDriver UP/DOWN classification benchmark")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--no-save", action="store_true", help="print results without writing files")
    args = parser.parse_args(argv)

    experiment = args.experiment if args.horizon == 1 else f"{args.experiment}_h{args.horizon}"
    result = run_evaluation(
        dataset_path=args.dataset,
        candidates=classification_candidates(),
        config=HarnessConfig(horizon=args.horizon, n_splits=args.n_splits),
        experiment=experiment,
    )
    print_report(result)

    if not args.no_save:
        paths = save_results(result, args.results_dir)
        print("\nSaved:")
        for p in paths.values():
            print(f"  {p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
