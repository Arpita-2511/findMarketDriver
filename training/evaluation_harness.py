"""
FindMarketDriver - central evaluation harness (Phase 1).

Every candidate model and feature set must be evaluated here before it
can be considered for production. See ARCHITECTURE.md section 9 and
REQUIREMENTS.md section 9 (REQ-EVAL-001 .. REQ-EVAL-010).

What it does
------------
1. Load the dataset and validate it (schema, ordering, duplicates,
   NaN / inf, target alignment).
2. Build both targets from Close:
       future_return = Close[t+h] / Close[t] - 1      (regression)
       direction     = 1 if future_return > 0 else 0  (classification)
3. Walk forward with TimeSeriesSplit(n_splits=20, gap=horizon).
4. For every fold, fit a FRESH clone of each candidate pipeline on the
   training fold only. Scaling, any feature selection and model fitting
   all live inside the pipeline, so no test row can influence them.
5. Pool the out-of-sample predictions from all folds and compute the
   benchmark metrics on them. Fold-level metrics are kept as well.
6. Compare every candidate with the strongest baseline of its track and
   mark it QUALIFIED only if it beats that baseline on a proper loss
   AND the improvement is statistically significant (one-sided
   Diebold-Mariano test). If nothing qualifies, the result is
   "NO QUALIFIED MODEL" - no winner is forced.
7. Save a JSON report, a CSV summary and the pooled OOS predictions to
   data/results/evaluation/.

Run from the project root:

    python -m training.evaluation_harness
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.feature_engineering import (  # noqa: E402
    TECHNICAL_FEATURE_VERSION,
    TECHNICAL_FEATURES,
)

DEFAULT_DATASET = PROJECT_ROOT / "data" / "final_stock_dataset.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "evaluation"

HARNESS_VERSION = "1.0"
GATE_VERSION = "gate_v1"

QUALIFIED = "QUALIFIED"
NOT_QUALIFIED = "NOT_QUALIFIED"
NO_QUALIFIED_MODEL = "NO QUALIFIED MODEL"

REGRESSION = "regression"
CLASSIFICATION = "classification"

# Log loss is undefined for probabilities of exactly 0 or 1 (e.g. the
# Always-UP baseline). Clip so the number is finite but still terrible.
PROBA_EPS = 1e-15


class DatasetValidationError(ValueError):
    """Raised when the dataset violates the evaluation data contract."""


# ==========================================
# Configuration
# ==========================================


@dataclass(frozen=True)
class HarnessConfig:
    """Settings for one evaluation run."""

    horizon: int = 1
    n_splits: int = 20
    significance_level: float = 0.05

    @property
    def gap(self) -> int:
        # gap = horizon so a training row's target window never overlaps
        # the first test row (REQ-EVAL-002)
        return self.horizon


@dataclass
class Candidate:
    """
    One model evaluated by the harness.

    `factory` must return a NEW unfitted estimator every time it is
    called; the harness additionally clones it per fold. Put scaling and
    feature selection inside a sklearn Pipeline so they are fitted on the
    training fold only.
    """

    name: str
    task: str  # REGRESSION or CLASSIFICATION
    factory: Callable[[], BaseEstimator]
    is_baseline: bool = False
    description: str = ""


# ==========================================
# Candidate registry
# ==========================================


def default_candidates() -> list[Candidate]:
    """
    Official Phase 1 candidates. Phase 3 adds tree models by appending
    Candidate entries here (or passing its own list) - the evaluation
    logic does not change.
    """
    return [
        # ----- regression baselines -----
        Candidate(
            "Mean Return", REGRESSION,
            lambda: DummyRegressor(strategy="mean"),
            is_baseline=True,
            description="Predicts the training-fold mean return",
        ),
        Candidate(
            "Zero Return", REGRESSION,
            lambda: DummyRegressor(strategy="constant", constant=0.0),
            is_baseline=True,
            description="Predicts 0% return (random walk)",
        ),
        # ----- regression models -----
        Candidate(
            "Linear Regression", REGRESSION,
            lambda: Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        ),
        Candidate(
            "Ridge", REGRESSION,
            lambda: Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
            description="alpha=1.0 (fixed; no tuning in Phase 1)",
        ),
        # ----- classification baselines -----
        Candidate(
            "Always UP", CLASSIFICATION,
            lambda: DummyClassifier(strategy="constant", constant=1),
            is_baseline=True,
            description="Always predicts UP with probability 1",
        ),
        Candidate(
            "Base Rate", CLASSIFICATION,
            lambda: DummyClassifier(strategy="prior"),
            is_baseline=True,
            description="P(UP) = training-fold share of UP days",
        ),
        # ----- classification models -----
        Candidate(
            "Logistic Regression", CLASSIFICATION,
            lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=1000)),
            ]),
            description="C=1.0 (fixed; no tuning in Phase 1)",
        ),
    ]


# ==========================================
# Data loading & validation
# ==========================================


def load_dataset(path: Path) -> pd.DataFrame:
    """Read the dataset CSV and parse the Date column."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise DatasetValidationError("Dataset has no 'Date' column")

    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    return df


def validate_dataset(df: pd.DataFrame, feature_columns: list[str]) -> dict:
    """
    Enforce the data contract. Raises DatasetValidationError listing
    every problem found; returns a summary dict when the data is valid.
    """
    problems: list[str] = []

    required = ["Date", "Close", *feature_columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"Missing required columns: {missing}")

    if not df["Date"].is_monotonic_increasing:
        problems.append("Dates are not in chronological order")

    n_dup_dates = int(df["Date"].duplicated().sum())
    if n_dup_dates:
        problems.append(f"{n_dup_dates} duplicate date(s)")

    checked = df[["Close", *feature_columns]]
    non_numeric = [c for c in checked.columns if not pd.api.types.is_numeric_dtype(checked[c])]
    if non_numeric:
        raise DatasetValidationError(f"Non-numeric columns: {non_numeric}")

    nan_cols = checked.columns[checked.isna().any()].tolist()
    if nan_cols:
        problems.append(f"NaN values in: {nan_cols}")

    inf_cols = checked.columns[np.isinf(checked).any()].tolist()
    if inf_cols:
        problems.append(f"Infinite values in: {inf_cols}")

    if (df["Close"] <= 0).any():
        problems.append("Non-positive Close prices")

    if problems:
        raise DatasetValidationError("; ".join(problems))

    return {
        "rows": int(len(df)),
        "first_date": df["Date"].iloc[0].date().isoformat(),
        "last_date": df["Date"].iloc[-1].date().isoformat(),
    }


def build_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Add `future_return` and `direction` built from Close, and drop the
    last `horizon` rows (their future is not in the dataset).

    If the dataset carries a stored `Target` column (next-day return from
    features/finalize_dataset.py) and horizon == 1, it must agree with the
    recomputed target - this is the target-alignment check.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    out = df.copy()
    out["future_return"] = out["Close"].shift(-horizon) / out["Close"] - 1

    if horizon == 1 and "Target" in out.columns:
        both = out["future_return"].notna()
        if not np.allclose(out.loc[both, "Target"], out.loc[both, "future_return"]):
            bad = int((~np.isclose(out.loc[both, "Target"], out.loc[both, "future_return"])).sum())
            raise DatasetValidationError(
                f"Stored Target disagrees with Close[t+1]/Close[t]-1 on {bad} row(s)"
            )

    out = out.iloc[:-horizon].reset_index(drop=True)
    out["direction"] = (out["future_return"] > 0).astype(int)
    return out


# ==========================================
# Metrics
# ==========================================


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, proba_up: np.ndarray) -> dict:
    """proba_up = predicted P(direction == 1). Hard label = proba >= 0.5."""
    p = np.clip(proba_up, PROBA_EPS, 1 - PROBA_EPS)
    labels = (proba_up >= 0.5).astype(int)
    # ROC-AUC is undefined when a fold contains a single class
    auc = float(roc_auc_score(y_true, proba_up)) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "Accuracy": float(accuracy_score(y_true, labels)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, labels)),
        "ROC_AUC": auc,
        "Brier": float(brier_score_loss(y_true, proba_up)),
        "Log_Loss": float(log_loss(y_true, p, labels=[0, 1])),
        "Share_Predicted_UP": float(labels.mean()),
    }


def per_observation_loss(task: str, y_true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Loss used by the qualification gate: squared error / log loss."""
    if task == REGRESSION:
        return (y_true - pred) ** 2
    p = np.clip(pred, PROBA_EPS, 1 - PROBA_EPS)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def diebold_mariano(loss_model: np.ndarray, loss_baseline: np.ndarray, horizon: int) -> dict:
    """
    One-sided Diebold-Mariano test of H0: model is not better than the
    baseline. d_t = loss_baseline - loss_model, so d > 0 means the model
    is better. Newey-West variance with (horizon - 1) lags accounts for
    overlapping multi-day targets.
    """
    d = np.asarray(loss_baseline, dtype=float) - np.asarray(loss_model, dtype=float)
    n = len(d)
    d_mean = d.mean()
    centered = d - d_mean

    long_run_var = np.dot(centered, centered) / n
    for lag in range(1, horizon):
        weight = 1 - lag / horizon
        long_run_var += 2 * weight * np.dot(centered[lag:], centered[:-lag]) / n

    if long_run_var <= 0:
        # identical losses -> no evidence of improvement
        return {"dm_stat": 0.0, "p_value": 1.0, "mean_loss_improvement": float(d_mean)}

    dm_stat = d_mean / np.sqrt(long_run_var / n)
    p_value = 1 - stats.norm.cdf(dm_stat)
    return {"dm_stat": float(dm_stat), "p_value": float(p_value), "mean_loss_improvement": float(d_mean)}


# ==========================================
# Walk-forward evaluation
# ==========================================


def make_splitter(config: HarnessConfig) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=config.n_splits, gap=config.gap)


def predict_output(estimator: BaseEstimator, task: str, X: pd.DataFrame) -> np.ndarray:
    """Regression -> predicted return. Classification -> P(UP)."""
    if task == REGRESSION:
        return np.asarray(estimator.predict(X), dtype=float)

    proba = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    if 1 not in classes:
        # training fold contained only DOWN days
        return np.zeros(len(X))
    return proba[:, classes.index(1)]


def walk_forward(
    data: pd.DataFrame,
    feature_columns: list[str],
    candidates: list[Candidate],
    config: HarnessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Run every candidate through the same chronological folds.

    Returns
    -------
    predictions : one row per (test observation, candidate)
    fold_metrics : one row per (fold, candidate)
    folds : fold boundaries (train/test index ranges and dates)
    """
    X = data[feature_columns]
    targets = {REGRESSION: data["future_return"].to_numpy(), CLASSIFICATION: data["direction"].to_numpy()}
    dates = data["Date"]

    pred_rows: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    folds: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(make_splitter(config).split(X), start=1):
        folds.append({
            "fold": fold,
            "train_start": dates.iloc[train_idx[0]].date().isoformat(),
            "train_end": dates.iloc[train_idx[-1]].date().isoformat(),
            "test_start": dates.iloc[test_idx[0]].date().isoformat(),
            "test_end": dates.iloc[test_idx[-1]].date().isoformat(),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        })

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

        for cand in candidates:
            y = targets[cand.task]
            estimator = clone(cand.factory())
            estimator.fit(X_train, y[train_idx])
            pred = predict_output(estimator, cand.task, X_test)

            pred_rows.append(pd.DataFrame({
                "row": test_idx,
                "Date": dates.iloc[test_idx].dt.date.astype(str).to_numpy(),
                "fold": fold,
                "model": cand.name,
                "task": cand.task,
                "y_true": y[test_idx],
                "prediction": pred,
            }))

            metric_fn = regression_metrics if cand.task == REGRESSION else classification_metrics
            fold_rows.append({"fold": fold, "model": cand.name, "task": cand.task,
                              **metric_fn(y[test_idx], pred)})

    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(fold_rows), folds


# ==========================================
# Pooled metrics & qualification gate
# ==========================================


def pooled_metrics(predictions: pd.DataFrame, candidates: list[Candidate]) -> dict[str, dict]:
    out = {}
    for cand in candidates:
        p = predictions[predictions["model"] == cand.name]
        metric_fn = regression_metrics if cand.task == REGRESSION else classification_metrics
        out[cand.name] = metric_fn(p["y_true"].to_numpy(), p["prediction"].to_numpy())
    return out


def qualify(
    predictions: pd.DataFrame,
    candidates: list[Candidate],
    metrics: dict[str, dict],
    config: HarnessConfig,
) -> dict[str, dict]:
    """
    Gate v1 (applied to non-baseline candidates; baselines are reported
    as BASELINE):

    Regression:
        reference = baseline with the lowest pooled MSE (Mean / Zero)
        QUALIFIED iff MSE < reference MSE
                  and one-sided DM p-value < significance_level

    Classification:
        reference = Base Rate (the probabilistic baseline)
        QUALIFIED iff Log Loss < reference Log Loss
                  and one-sided DM p-value (per-observation log loss)
                      < significance_level
                  and Accuracy >= Always UP accuracy
                      (a direction product must not lose to "always UP")
    """
    results: dict[str, dict] = {}
    by_task = {t: [c for c in candidates if c.task == t] for t in (REGRESSION, CLASSIFICATION)}

    for task, cands in by_task.items():
        baselines = [c for c in cands if c.is_baseline]
        models = [c for c in cands if not c.is_baseline]
        if not models:
            continue
        if not baselines:
            raise ValueError(f"No baseline registered for task '{task}'")

        if task == REGRESSION:
            reference = min(baselines, key=lambda c: metrics[c.name]["MSE"])
            primary = "MSE"
        else:
            probabilistic = [c for c in baselines if c.name == "Base Rate"] or baselines
            reference = min(probabilistic, key=lambda c: metrics[c.name]["Log_Loss"])
            primary = "Log_Loss"
            always_up = next((c for c in baselines if c.name == "Always UP"), None)

        ref_pred = predictions[predictions["model"] == reference.name].sort_values("row")
        y = ref_pred["y_true"].to_numpy()
        ref_loss = per_observation_loss(task, y, ref_pred["prediction"].to_numpy())

        for cand in baselines:
            results[cand.name] = {"status": "BASELINE", "task": task}

        for cand in models:
            cand_pred = predictions[predictions["model"] == cand.name].sort_values("row")
            if not np.array_equal(cand_pred["row"].to_numpy(), ref_pred["row"].to_numpy()):
                raise RuntimeError(f"{cand.name} and {reference.name} were scored on different rows")

            cand_loss = per_observation_loss(task, y, cand_pred["prediction"].to_numpy())
            dm = diebold_mariano(cand_loss, ref_loss, config.horizon)

            better = metrics[cand.name][primary] < metrics[reference.name][primary]
            significant = dm["p_value"] < config.significance_level
            reasons = []
            if not better:
                reasons.append(f"{primary} not better than {reference.name}")
            elif not significant:
                reasons.append(f"improvement over {reference.name} not significant "
                               f"(p={dm['p_value']:.3f})")

            if task == CLASSIFICATION and always_up is not None:
                if metrics[cand.name]["Accuracy"] < metrics[always_up.name]["Accuracy"]:
                    reasons.append("Accuracy below Always UP")

            results[cand.name] = {
                "status": QUALIFIED if not reasons else NOT_QUALIFIED,
                "task": task,
                "reference_baseline": reference.name,
                "primary_metric": primary,
                "model_value": metrics[cand.name][primary],
                "baseline_value": metrics[reference.name][primary],
                "diebold_mariano": dm,
                "reasons": reasons,
            }

    return results


# ==========================================
# Provenance
# ==========================================


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip() != ""
        return {"commit": commit, "uncommitted_changes": dirty}
    except (OSError, subprocess.CalledProcessError) as e:
        return {"commit": None, "error": str(e)}


def environment_info() -> dict:
    packages = {}
    for pkg in ["numpy", "pandas", "scikit-learn", "scipy", "yfinance"]:
        try:
            packages[pkg] = version(pkg)
        except PackageNotFoundError:
            packages[pkg] = None
    return {"python": platform.python_version(), "packages": packages}


# ==========================================
# Orchestration
# ==========================================


@dataclass
class EvaluationResult:
    report: dict
    summary: pd.DataFrame
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame = field(repr=False)


def run_evaluation(
    dataset_path: Path = DEFAULT_DATASET,
    feature_columns: list[str] | None = None,
    feature_version: str = TECHNICAL_FEATURE_VERSION,
    candidates: list[Candidate] | None = None,
    config: HarnessConfig | None = None,
    experiment: str = "technical_baseline",
    data: pd.DataFrame | None = None,
) -> EvaluationResult:
    """
    Evaluate every candidate on one dataset. Pass `data` to evaluate an
    in-memory frame (used by tests); otherwise `dataset_path` is loaded.
    """
    config = config or HarnessConfig()
    feature_columns = list(feature_columns or TECHNICAL_FEATURES)
    candidates = candidates or default_candidates()

    if data is None:
        dataset_path = Path(dataset_path)
        df = load_dataset(dataset_path)
        dataset_version = {
            "path": dataset_path.relative_to(PROJECT_ROOT).as_posix()
            if dataset_path.is_relative_to(PROJECT_ROOT) else str(dataset_path),
            "sha256": file_sha256(dataset_path),
        }
    else:
        df = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"], utc=True)
        dataset_version = {"path": None, "sha256": None, "note": "in-memory data"}

    dataset_summary = validate_dataset(df, feature_columns)
    dataset_version.update(dataset_summary)

    evaluable = build_targets(df, config.horizon)
    predictions, fold_metrics, folds = walk_forward(evaluable, feature_columns, candidates, config)

    # every evaluated row must be predicted exactly once per candidate
    counts = predictions.groupby("model")["row"].agg(["count", "nunique"])
    if not (counts["count"] == counts["nunique"]).all():
        raise RuntimeError("A test row was predicted more than once for the same model")

    metrics = pooled_metrics(predictions, candidates)
    gate = qualify(predictions, candidates, metrics, config)

    qualified = {
        task: sorted(n for n, g in gate.items() if g["task"] == task and g["status"] == QUALIFIED)
        for task in (REGRESSION, CLASSIFICATION)
    }
    overall = {task: (names if names else NO_QUALIFIED_MODEL) for task, names in qualified.items()}

    summary = pd.DataFrame([
        {"experiment": experiment, "model": c.name, "task": c.task,
         "is_baseline": c.is_baseline, "status": gate[c.name]["status"],
         **{k: v for k, v in metrics[c.name].items()}}
        for c in candidates
    ])

    target_definitions = {
        REGRESSION: f"future_return = Close[t+{config.horizon}] / Close[t] - 1",
        CLASSIFICATION: "direction = 1 if future_return > 0 else 0",
    }

    report = {
        "experiment": experiment,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "harness_version": HARNESS_VERSION,
        "gate_version": GATE_VERSION,
        "dataset_version": dataset_version,
        "feature_version": feature_version,
        "features": feature_columns,
        "target": target_definitions,
        "horizon_days": config.horizon,
        "validation_method": "TimeSeriesSplit walk-forward, pooled out-of-sample",
        "n_splits": config.n_splits,
        "gap": config.gap,
        "significance_level": config.significance_level,
        "training_period": {"start": folds[0]["train_start"], "end": folds[-1]["train_end"]},
        "evaluation_period": {"start": folds[0]["test_start"], "end": folds[-1]["test_end"],
                              "n_oos_rows": int(sum(f["n_test"] for f in folds))},
        "qualification_result": overall,
        "models": [
            {
                "model": c.name,
                "task": c.task,
                "is_baseline": c.is_baseline,
                "description": c.description,
                "metrics": metrics[c.name],
                "qualification": gate[c.name],
            }
            for c in candidates
        ],
        "baseline_metrics": {c.name: metrics[c.name] for c in candidates if c.is_baseline},
        "folds": folds,
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "git": git_info(),
        "environment": environment_info(),
    }

    return EvaluationResult(report=report, summary=summary, predictions=predictions,
                            fold_metrics=fold_metrics)


def _json_safe(obj):
    """Convert numpy scalars to Python types and NaN/inf to None (strict JSON)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def save_results(result: EvaluationResult, results_dir: Path = DEFAULT_RESULTS_DIR) -> dict[str, Path]:
    """Write <experiment>_<YYYYMMDD>.json / .csv / _predictions.csv."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.fromisoformat(result.report["timestamp"]).astimezone().strftime("%Y%m%d")
    stem = f"{result.report['experiment']}_{stamp}"

    paths = {
        "report": results_dir / f"{stem}.json",
        "summary": results_dir / f"{stem}.csv",
        "predictions": results_dir / f"{stem}_predictions.csv",
    }
    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump(_json_safe(result.report), f, indent=2, allow_nan=False)
    result.summary.to_csv(paths["summary"], index=False)
    result.predictions.to_csv(paths["predictions"], index=False)
    return paths


def print_report(result: EvaluationResult) -> None:
    r = result.report
    print("=" * 78)
    print(f"EVALUATION: {r['experiment']}  ({r['feature_version']}, horizon={r['horizon_days']}d)")
    print("=" * 78)
    dv = r["dataset_version"]
    print(f"Dataset   : {dv.get('path')}  rows={dv['rows']}  {dv['first_date']} -> {dv['last_date']}")
    print(f"Validation: TimeSeriesSplit(n_splits={r['n_splits']}, gap={r['gap']}), "
          f"{r['evaluation_period']['n_oos_rows']} pooled OOS rows "
          f"({r['evaluation_period']['start']} -> {r['evaluation_period']['end']})")

    s = result.summary
    reg = s[s["task"] == REGRESSION][["model", "status", "MAE", "RMSE", "R2"]]
    clf = s[s["task"] == CLASSIFICATION][["model", "status", "Accuracy", "Balanced_Accuracy",
                                          "ROC_AUC", "Brier", "Log_Loss"]]
    print("\n-- Regression (pooled OOS) --")
    print(reg.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\n-- Classification (pooled OOS) --")
    print(clf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n-- Qualification gate --")
    for m in r["models"]:
        q = m["qualification"]
        if q["status"] == "BASELINE":
            continue
        dm = q["diebold_mariano"]
        why = "; ".join(q["reasons"]) or "beats baseline significantly"
        print(f"{m['model']:20s} {q['status']:14s} vs {q['reference_baseline']:12s} "
              f"DM p={dm['p_value']:.3f}  ({why})")
    print("\nResult:")
    for task, res in r["qualification_result"].items():
        print(f"  {task:15s}: {res}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FindMarketDriver evaluation harness")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--experiment", default="technical_baseline")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--no-save", action="store_true", help="print results without writing files")
    args = parser.parse_args(argv)

    result = run_evaluation(
        dataset_path=args.dataset,
        config=HarnessConfig(horizon=args.horizon, n_splits=args.n_splits),
        experiment=args.experiment,
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
