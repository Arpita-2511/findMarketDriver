"""
Phase 9 - predictive signal research (AAPL), pre-registered two-stage design.

    freeze            persist the complete experiment matrix (feature sets,
                      horizons, targets, models + fixed parameters, evaluation
                      configuration) BEFORE any development result exists
    validate-context  build the research frame from the verified inputs and
                      the SPY/QQQ snapshots; read-only report
    develop           run every pre-registered experiment on the DEVELOPMENT
                      period (sessions <= 2024-09-30) through the UNCHANGED
                      central harness; Holm-adjusted p-values (informational);
                      freeze the development qualifiers
    holdout --confirm run each frozen development qualifier exactly once on the
                      Phase-9 confirmation holdout (2024-10-01 .. 2026-09-25)
                      with the same gate; with no qualifiers, records
                      NO DEVELOPMENT QUALIFIERS and reads no data

Nothing here changes gate_v1, the baselines, the target formulas, the leakage
rule or any Phase 7/8 artifact. A model is CONFIRMED only if it is QUALIFIED
in development AND on the holdout.

Holdout procedure (same walk-forward semantics as the harness, test region
restricted to the holdout): the labelled holdout rows are cut into n_splits
contiguous folds; each fold is predicted by a fresh model trained on ALL
earlier rows except the last `horizon` (gap = horizon, as TimeSeriesSplit).
Pooled holdout predictions -> the harness's pooled_metrics and qualify().
The holdout is NOT a pristine test set: Phase 3 (technical trees, h = 1) and
Phase 8 (linear A/B/C, h = 1) already evaluated parts of it.

Usage (project root):
    python -m training.phase9_research freeze
    python -m training.phase9_research validate-context --spy <SPY csv> --qqq <QQQ csv>
    python -m training.phase9_research develop --spy <SPY csv> --qqq <QQQ csv>
    python -m training.phase9_research holdout --spy <SPY csv> --qqq <QQQ csv> --confirm
    python -m training.phase9_research status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lightgbm import LGBMRegressor  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from xgboost import XGBRegressor  # noqa: E402

from services import market_context as mc  # noqa: E402
from services import sentiment_window as sw  # noqa: E402
from services.historical_sentiment import load_sentiment_dataset  # noqa: E402
from services.market_calendar_service import TradingCalendar  # noqa: E402
from services.news_schema import parse_timestamp  # noqa: E402
from training.build_dataset import file_sha256, load_raw_snapshot, meta_path_for  # noqa: E402
from training.evaluation_harness import (  # noqa: E402
    CLASSIFICATION,
    GATE_VERSION,
    HARNESS_VERSION,
    QUALIFIED,
    REGRESSION,
    Candidate,
    HarnessConfig,
    _json_safe,
    build_targets,
    classification_metrics,
    default_candidates,
    git_info,
    pooled_metrics,
    predict_output,
    qualify,
    regression_metrics,
    run_evaluation,
    validate_dataset,
)
from training.feature_store import (  # noqa: E402
    EVENT_NUMERIC_COLUMNS,
    SENTIMENT_COLUMNS,
    TECHNICAL_COLUMNS,
    load_feature_store,
)
from training.incremental_evaluation import evaluation_frame  # noqa: E402
from training.research_targets import HORIZONS, target_definition  # noqa: E402
from training.train_classification import SEED, classification_candidates  # noqa: E402

MATRIX_VERSION = "phase9_matrix_v1"
REGISTRY_VERSION = "phase9_registry_v1"
SYMBOL = "AAPL"
N_SPLITS = 20
DEV_END = date(2024, 9, 30)
HOLDOUT_START = date(2024, 10, 1)
HOLDOUT_END = date(2026, 9, 25)

RESEARCH_DIR = PROJECT_ROOT / "data" / "results" / "research" / "phase9"
REGISTRY_PATH = RESEARCH_DIR / "registry.json"
DEFAULT_STORE = (PROJECT_ROOT / "data" / "processed" / "features" / "backfill_2017_2026"
                 / "feature_store_v1_AAPL_20170126_20260928.csv")
SENTIMENT_DIR = PROJECT_ROOT / "data" / "processed" / "sentiment"
EVENTS_DIR = PROJECT_ROOT / "data" / "processed" / "events"
NEWS_DIR = PROJECT_ROOT / "data" / "processed" / "news"
RAW_STOCKS_DIR = PROJECT_ROOT / "data" / "raw" / "stocks"

NO_DEVELOPMENT_QUALIFIERS = "NO DEVELOPMENT QUALIFIERS"
NO_CONFIRMED_MODEL = "NO CONFIRMED MODEL"


class ResearchError(ValueError):
    """The research protocol would be violated (changed matrix, reused stage, bad input)."""


# ==========================================
# Pre-registered candidates (fixed; no tuning)
# ==========================================


def regression_candidates() -> list[Candidate]:
    """Harness regression baselines + Linear/Ridge (unchanged) + fixed, regularized trees."""
    base = [c for c in default_candidates() if c.task == REGRESSION]          # Mean, Zero, Linear, Ridge
    trees = [
        Candidate("RF Regressor", REGRESSION,
                  lambda: RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=50,
                                                max_features="sqrt", random_state=SEED, n_jobs=1),
                  description="Phase 3 Random Forest settings, regression objective (fixed)"),
        Candidate("HistGradientBoosting Regressor", REGRESSION,
                  lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=3,
                                                        max_leaf_nodes=7, min_samples_leaf=50,
                                                        l2_regularization=1.0, early_stopping=False,
                                                        random_state=SEED),
                  description="300 iter, depth 3 / 7 leaves, lr 0.03, min leaf 50, no early stopping "
                              "(its internal split would not be temporal)"),
        Candidate("XGBoost Regressor", REGRESSION,
                  lambda: XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                                       colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
                                       objective="reg:squarederror", tree_method="hist",
                                       random_state=SEED, n_jobs=1),
                  description="Phase 3 XGBoost settings, squared-error objective (fixed)"),
        Candidate("LightGBM Regressor", REGRESSION,
                  lambda: LGBMRegressor(n_estimators=300, max_depth=3, num_leaves=7, learning_rate=0.03,
                                        min_child_samples=50, subsample=0.8, subsample_freq=1,
                                        colsample_bytree=0.8, random_state=SEED, n_jobs=1,
                                        deterministic=True, verbose=-1),
                  description="Phase 3 LightGBM settings, regression objective (fixed)"),
    ]
    return base + trees


def phase9_classification_candidates() -> list[Candidate]:
    """Phase 3 classification candidates (baselines, Logistic, RF, XGBoost, LightGBM) + HistGradientBoosting."""
    return classification_candidates() + [
        Candidate("HistGradientBoosting", CLASSIFICATION,
                  lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=3,
                                                         max_leaf_nodes=7, min_samples_leaf=50,
                                                         l2_regularization=1.0, early_stopping=False,
                                                         random_state=SEED),
                  description="300 iter, depth 3 / 7 leaves, lr 0.03, min leaf 50, no early stopping"),
    ]


def phase9_candidates() -> list[Candidate]:
    return regression_candidates() + phase9_classification_candidates()


# ==========================================
# Feature sets
# ==========================================

SENTIMENT_WINDOW_COLUMNS = tuple(sw.COLUMNS)
MARKET_CONTEXT_COLUMNS = tuple(mc.COLUMNS)

FEATURE_SETS = {
    "A": {"families": ["technical"], "columns": list(TECHNICAL_COLUMNS)},
    "B": {"families": ["technical", "sentiment"], "columns": [*TECHNICAL_COLUMNS, *SENTIMENT_COLUMNS]},
    "C": {"families": ["technical", "events"], "columns": [*TECHNICAL_COLUMNS, *EVENT_NUMERIC_COLUMNS]},
    "D": {"families": ["technical", "sentiment", "events"],
          "columns": [*TECHNICAL_COLUMNS, *SENTIMENT_COLUMNS, *EVENT_NUMERIC_COLUMNS]},
    "E": {"families": ["technical", "market_context"], "columns": [*TECHNICAL_COLUMNS, *MARKET_CONTEXT_COLUMNS]},
    "F": {"families": ["technical", "sentiment_window", "events", "market_context"],
          "columns": [*TECHNICAL_COLUMNS, *SENTIMENT_WINDOW_COLUMNS, *EVENT_NUMERIC_COLUMNS,
                      *MARKET_CONTEXT_COLUMNS]},
}
ALL_FEATURE_COLUMNS = list(dict.fromkeys(c for fs in FEATURE_SETS.values() for c in fs["columns"]))


# ==========================================
# Matrix (pre-registration)
# ==========================================


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def candidate_spec(candidate: Candidate) -> dict:
    est = candidate.factory()
    params = _jsonable(est.get_params(deep=False))
    seed = params.get("random_state") if isinstance(params, dict) else None
    return {"name": candidate.name, "task": candidate.task, "is_baseline": candidate.is_baseline,
            "description": candidate.description,
            "class": f"{type(est).__module__}.{type(est).__name__}", "params": params,
            "random_seed": seed if isinstance(seed, int) else None}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def experiment_id(feature_set: str, horizon: int, model: str) -> str:
    return f"p9_{feature_set}_h{horizon}_{_slug(model)}"


@dataclass(frozen=True)
class ResearchSpec:
    """Everything that defines the experiment matrix. OFFICIAL is the Phase 9 pre-registration."""

    feature_sets: dict = field(default_factory=lambda: FEATURE_SETS)
    horizons: tuple = HORIZONS
    candidates_factory: Callable[[], list[Candidate]] = phase9_candidates
    n_splits: int = N_SPLITS
    dev_end: date = DEV_END
    holdout_start: date = HOLDOUT_START
    holdout_end: date = HOLDOUT_END

    def candidates(self) -> list[Candidate]:
        cands = self.candidates_factory()
        names = [c.name for c in cands]
        if len(set(names)) != len(names):
            raise ResearchError(f"candidate names must be unique (the harness keys results by name): {names}")
        return cands

    def matrix(self) -> dict:
        cands = self.candidates()
        significance = HarnessConfig().significance_level
        experiments = [{"experiment_id": experiment_id(fs, h, c.name), "feature_set": fs, "horizon": h,
                        "model": c.name, "task": c.task}
                       for fs in self.feature_sets for h in self.horizons for c in cands if not c.is_baseline]
        return {
            "matrix_version": MATRIX_VERSION,
            "symbol": SYMBOL,
            "feature_sets": {k: {"families": list(v["families"]), "columns": list(v["columns"]),
                                 "n_features": len(v["columns"])} for k, v in self.feature_sets.items()},
            "feature_definitions": {
                "technical": "technical_v2 (unchanged)",
                "sentiment": "sentiment_features_v1 (unchanged, cumulative), prefix sentiment__",
                "events": "event_features_v1 numeric (unchanged), prefix event__",
                sw.SENTIMENT_WINDOW_VERSION: sw.DEFINITIONS,
                mc.MARKET_CONTEXT_VERSION: mc.DEFINITIONS,
            },
            "horizons": list(self.horizons),
            "targets": {str(h): target_definition(h) for h in self.horizons},
            "candidates": [candidate_spec(c) for c in cands],
            "evaluation": {
                "harness": "training.evaluation_harness.run_evaluation (unchanged)",
                "harness_version": HARNESS_VERSION,
                "gate_version": GATE_VERSION,
                "validation": "TimeSeriesSplit walk-forward, pooled out-of-sample",
                "n_splits": self.n_splits,
                "gap": "horizon",
                "significance_level": significance,
                "baselines": {"regression": ["Mean Return", "Zero Return"],
                              "classification": ["Always UP", "Base Rate"]},
                "tuning": "none - every parameter fixed in this matrix",
                "row_selection": ("Phase 8 feature-store rows covered by every family, incl. sentiment_window_v1 "
                                  "(20-session window inside the news interval) and market_context_v1 (strict); "
                                  "identical rows for every feature set; targets rebuilt from Close by the harness"),
                "development_period": {"end": self.dev_end.isoformat(),
                                       "rule": "rows with trading_date <= end; targets built inside this frame, "
                                               "so no holdout price is used"},
                "holdout_period": {"name": "Phase-9 confirmation holdout", "start": self.holdout_start.isoformat(),
                                   "end": self.holdout_end.isoformat(),
                                   "limitation": "not pristine: Phase 3 / Phase 8 evaluated parts of it"},
                "holdout_procedure": ("only frozen development qualifiers, each exactly once; labelled holdout rows in "
                                      f"{self.n_splits} contiguous folds, expanding-window training on all earlier rows "
                                      "minus a gap of `horizon` rows; same metrics and gate_v1"),
                "confirmation_rule": "CONFIRMED iff QUALIFIED in development AND on the holdout",
                "multiple_comparisons": {"method": "Holm", "scope": "all development non-baseline tests",
                                         "role": "informational only; gate_v1 is authoritative"},
            },
            "experiments": experiments,
        }


OFFICIAL = ResearchSpec()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def matrix_sha256(matrix: dict) -> str:
    return hashlib.sha256(canonical_json(matrix).encode("utf-8")).hexdigest()


# ==========================================
# Multiple comparisons (informational)
# ==========================================


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in input order (ties keep input order)."""
    p = [float(x) for x in p_values]
    if any(not (0.0 <= x <= 1.0) for x in p):
        raise ValueError("p-values must lie in [0, 1]")
    m = len(p)
    order = sorted(range(m), key=lambda i: (p[i], i))
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted


# ==========================================
# Provenance
# ==========================================


def environment() -> dict:
    pkgs = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "xgboost", "lightgbm"):
        try:
            pkgs[name] = version(name)
        except PackageNotFoundError:
            pkgs[name] = None
    return {"python": platform.python_version(), "packages": pkgs}


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)


def _verified(path: Path, expected_sha: str | None = None) -> dict:
    """File sha256 must equal its .meta.json sha256 (and `expected_sha` if given)."""
    path = Path(path)
    meta = json.loads(meta_path_for(path).read_text(encoding="utf-8"))
    actual = file_sha256(path)
    if actual != meta.get("sha256"):
        raise ResearchError(f"{path.name}: sha256 does not match its metadata")
    if expected_sha is not None and actual != expected_sha:
        raise ResearchError(f"{path.name}: sha256 differs from the hash recorded upstream")
    return {"path": _rel(path), "sha256": actual}


def input_provenance(store_path: Path = DEFAULT_STORE) -> dict:
    """Locate and hash-verify every upstream artifact of the feature store (full chain)."""
    store_path = Path(store_path)
    store = _verified(store_path)
    smeta = json.loads(meta_path_for(store_path).read_text(encoding="utf-8"))
    src = smeta["sources"]
    s_daily = SENTIMENT_DIR / src["sentiment"]["file"]
    e_daily = EVENTS_DIR / src["events"]["file"]
    snapshot = RAW_STOCKS_DIR / src["calendar"]["file"]
    out = {"feature_store": store,
           "sentiment_daily": _verified(s_daily, src["sentiment"]["sha256"]),
           "event_daily": _verified(e_daily, src["events"]["sha256"]),
           "aapl_snapshot": _verified(snapshot, src["calendar"]["sha256"])}
    sdm = json.loads(meta_path_for(s_daily).read_text(encoding="utf-8"))
    records = SENTIMENT_DIR / sdm["source_sentiment"]["file"]
    out["sentiment_records"] = _verified(records, sdm["source_sentiment"]["sha256"])
    rmeta = json.loads(meta_path_for(records).read_text(encoding="utf-8"))
    canonical = NEWS_DIR / rmeta["source_dataset"]["file"]
    out["canonical_news"] = _verified(canonical, rmeta["source_dataset"]["sha256"])
    return out


def _snapshot_info(path: Path, meta: dict) -> dict:
    return {"path": _rel(path), "sha256": meta["sha256"], "ticker": meta.get("ticker"),
            "retrieved_at_utc": meta.get("retrieved_at_utc"), "first_date": meta.get("first_date"),
            "last_date": meta.get("last_date"), "rows": meta.get("rows")}


# ==========================================
# Research frame
# ==========================================


def build_research_frame(store_path: Path, spy_path: Path, qqq_path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Phase 8 store rows (Date, Close, Target, technical, sentiment, events) +
    sentiment_window_v1 + market_context_v1, restricted to rows where every
    family is defined. Read-only on every input.
    """
    inputs = input_provenance(store_path)
    store, _ = load_feature_store(store_path)
    base = evaluation_frame(store)
    bars, _ = load_raw_snapshot(PROJECT_ROOT / inputs["aapl_snapshot"]["path"])
    calendar = TradingCalendar.from_bars(bars)

    canonical = PROJECT_ROOT / inputs["canonical_news"]["path"]
    scored, _ = load_sentiment_dataset(PROJECT_ROOT / inputs["sentiment_records"]["path"], canonical)
    cmeta = json.loads(meta_path_for(canonical).read_text(encoding="utf-8"))
    dates = [ts.date() for ts in base["Date"]]
    windows = sw.generate_sentiment_window_features(
        scored, dates, calendar, SYMBOL, coverage_start=parse_timestamp(cmeta["start"], "start"),
        coverage_end=parse_timestamp(cmeta["end"], "end"))
    covered = windows["covered"].to_numpy()
    if covered.any():
        first = int(np.argmax(covered))
        if not covered[first:].all():
            raise ResearchError("sentiment-window coverage has a gap after its first covered row")
    else:
        raise ResearchError("no row has a fully covered 20-session sentiment window")
    windows = windows[covered].reset_index(drop=True)
    keep_dates = list(windows["trading_date"])

    spy_bars, spy_meta = mc.load_context_snapshot(spy_path, "SPY")
    qqq_bars, qqq_meta = mc.load_context_snapshot(qqq_path, "QQQ")
    context = mc.compute_market_context(bars, spy_bars, qqq_bars, keep_dates)

    keep = set(keep_dates)
    frame = base[[d in keep for d in dates]].reset_index(drop=True)
    if [ts.date() for ts in frame["Date"]] != keep_dates or list(context["trading_date"]) != keep_dates:
        raise ResearchError("internal error: research rows are misaligned")
    if not (windows["prediction_timestamp"].equals(context["prediction_timestamp"])):
        raise ResearchError("internal error: sentiment-window and market-context timestamps differ")
    for c in SENTIMENT_WINDOW_COLUMNS:
        frame[c] = windows[c].to_numpy()
    for c in MARKET_CONTEXT_COLUMNS:
        frame[c] = context[c].to_numpy()
    values = frame[ALL_FEATURE_COLUMNS].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ResearchError("research frame contains NaN or infinite feature values")

    provenance = {
        "inputs": inputs,
        "spy_snapshot": _snapshot_info(spy_path, spy_meta),
        "qqq_snapshot": _snapshot_info(qqq_path, qqq_meta),
        "frame": {"rows": len(frame), "first_date": keep_dates[0].isoformat(),
                  "last_date": keep_dates[-1].isoformat(), "store_evaluable_rows": len(base),
                  "rows_before_full_sentiment_window": len(base) - len(frame),
                  "n_feature_columns": len(ALL_FEATURE_COLUMNS)},
    }
    return frame, provenance


# ==========================================
# Registry I/O
# ==========================================


def _write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_json_safe(obj), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def _write_new_json(path: Path, obj) -> dict:
    """Write-once artifact; returns {path, sha256}."""
    path = Path(path)
    if path.exists():
        raise ResearchError(f"{path.name} already exists; research artifacts are never overwritten")
    _write_json_atomic(path, obj)
    return {"path": _rel(path), "sha256": file_sha256(path)}


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ResearchError(f"registry not found: {path} (run `freeze` first)")
    reg = json.loads(path.read_text(encoding="utf-8"))
    if reg.get("registry_version") != REGISTRY_VERSION:
        raise ResearchError(f"unsupported registry version {reg.get('registry_version')!r}")
    if matrix_sha256(reg["matrix"]) != reg["matrix_sha256"]:
        raise ResearchError("registry matrix does not match its recorded hash (edited after freezing)")
    return reg


def _check_spec(reg: dict, spec: ResearchSpec) -> dict:
    matrix = spec.matrix()
    if matrix_sha256(matrix) != reg["matrix_sha256"]:
        raise ResearchError("the experiment matrix in code differs from the FROZEN matrix; "
                            "pre-registered experiments cannot be changed")
    return matrix


def freeze(registry_path: Path = REGISTRY_PATH, *, inputs: dict, spec: ResearchSpec = OFFICIAL) -> dict:
    """Persist the pre-registered matrix. Refuses to overwrite an existing registry."""
    registry_path = Path(registry_path)
    if registry_path.exists():
        raise ResearchError(f"{registry_path} already exists; the matrix is frozen once")
    matrix = spec.matrix()
    reg = {
        "registry_version": REGISTRY_VERSION,
        "matrix": matrix,
        "matrix_sha256": matrix_sha256(matrix),
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_git": git_info(),
        "frozen_environment": environment(),
        "inputs": inputs,
        "status": "FROZEN",
        "development": None,
        "holdout": None,
        "results": {e["experiment_id"]: {**e, "status": "PENDING", "development": None, "holdout": None}
                    for e in matrix["experiments"]},
    }
    _write_json_atomic(registry_path, reg)
    return reg


# ==========================================
# Development stage
# ==========================================


def development_frame(frame: pd.DataFrame, dev_end: date) -> pd.DataFrame:
    """Rows with trading_date <= dev_end. Targets are later built INSIDE this frame (no holdout prices)."""
    dev = frame[frame["Date"].dt.date <= dev_end].reset_index(drop=True)
    if dev.empty or dev["Date"].iloc[-1].date() > dev_end:
        raise ResearchError("development frame is empty or extends past the development end")
    return dev


def _summaries(report: dict) -> tuple[dict, dict]:
    models = {m["model"]: m for m in report["models"]}
    baselines = {task: {n: m["metrics"] for n, m in models.items() if m["is_baseline"] and m["task"] == task}
                 for task in (REGRESSION, CLASSIFICATION)}
    return models, baselines


def run_development(registry_path: Path, frame: pd.DataFrame, context_provenance: dict, *,
                    spec: ResearchSpec = OFFICIAL, results_dir: Path | None = None,
                    run_eval=run_evaluation, log=print) -> dict:
    registry_path = Path(registry_path)
    reg = load_registry(registry_path)
    if reg["status"] != "FROZEN" or reg["development"] is not None:
        raise ResearchError("development has already been run for this registry")
    matrix = _check_spec(reg, spec)
    results_dir = Path(results_dir) if results_dir is not None else registry_path.parent
    exp_dir = results_dir / "experiments"
    planned = [exp_dir / f"dev_{fs}_h{h}.json" for fs in spec.feature_sets for h in spec.horizons]
    if any(p.exists() for p in planned):
        raise ResearchError(f"development artifacts already exist in {exp_dir}")

    dev = development_frame(frame, spec.dev_end)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    specs = {c["name"]: c for c in matrix["candidates"]}
    entries, runs = [], []
    for fs, fdef in spec.feature_sets.items():
        cols = list(fdef["columns"])
        for h in spec.horizons:
            name = f"phase9_dev_{fs}_h{h}"
            log(f"[develop] {name}: {len(cols)} features, horizon {h} ...")
            data = dev[["Date", "Close", *cols]].copy()
            result = run_eval(data=data, feature_columns=cols, feature_version=f"phase9_{fs}",
                              candidates=spec.candidates(), config=HarnessConfig(horizon=h, n_splits=spec.n_splits),
                              experiment=name)
            report = result.report
            artifact = _write_new_json(exp_dir / f"dev_{fs}_h{h}.json", report)
            models, baselines = _summaries(report)
            runs.append({"feature_set": fs, "horizon": h, "artifact": artifact,
                         "evaluation_period": report["evaluation_period"],
                         "training_period": report["training_period"],
                         "qualification_result": report["qualification_result"],
                         "baseline_metrics": baselines})
            for mname, m in models.items():
                if m["is_baseline"]:
                    continue
                q = m["qualification"]
                entries.append({
                    "experiment_id": experiment_id(fs, h, mname), "feature_set": fs, "horizon": h,
                    "model": mname, "task": m["task"],
                    "target": matrix["targets"][str(h)][m["task"]],
                    "parameters": specs[mname]["params"], "random_seed": specs[mname]["random_seed"],
                    "development": {
                        "frame_period": [dev["Date"].iloc[0].date().isoformat(),
                                         dev["Date"].iloc[-1].date().isoformat()],
                        "oos_period": report["evaluation_period"], "training_period": report["training_period"],
                        "n_oos_rows": report["evaluation_period"]["n_oos_rows"],
                        "n_splits": report["n_splits"], "gap": report["gap"],
                        "metrics": m["metrics"], "baseline_metrics": baselines[m["task"]],
                        "gate": q, "status": q["status"],
                        "p_value_raw": q["diebold_mariano"]["p_value"], "p_value_holm": None,
                        "artifact": artifact,
                    },
                })

    planned_ids = [e["experiment_id"] for e in matrix["experiments"]]
    if sorted(e["experiment_id"] for e in entries) != sorted(planned_ids):
        raise ResearchError("development results do not cover exactly the pre-registered experiments")
    holm = holm_adjust([e["development"]["p_value_raw"] for e in entries])
    for e, p in zip(entries, holm):
        e["development"]["p_value_holm"] = p
    qualifiers = [e["experiment_id"] for e in entries if e["development"]["status"] == QUALIFIED]

    for e in entries:
        reg["results"][e["experiment_id"]].update(
            {"status": "DEVELOPMENT_" + e["development"]["status"], **{k: v for k, v in e.items()}})
    reg["development"] = {
        "status": "COMPLETE", "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(), "environment": environment(), "context_inputs": context_provenance,
        "frame": {"rows": len(dev), "first_date": dev["Date"].iloc[0].date().isoformat(),
                  "last_date": dev["Date"].iloc[-1].date().isoformat()},
        "runs": runs, "n_tests": len(entries),
        "multiple_comparisons": {"method": "Holm", "n_tests": len(entries),
                                 "n_raw_p_below_alpha": sum(e["development"]["p_value_raw"] < 0.05 for e in entries),
                                 "n_holm_p_below_alpha": sum(p < 0.05 for p in holm),
                                 "role": "informational; gate_v1 decides qualification"},
        "qualifiers": qualifiers,
        "result": (f"{len(qualifiers)} DEVELOPMENT QUALIFIER(S)" if qualifiers else NO_DEVELOPMENT_QUALIFIERS),
    }
    reg["status"] = "DEVELOPMENT_COMPLETE"
    _write_json_atomic(registry_path, reg)
    return reg


# ==========================================
# Confirmation holdout
# ==========================================


def walk_forward_with_splits(data: pd.DataFrame, feature_columns: list[str], candidates: list[Candidate],
                             splits: list[tuple[np.ndarray, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    The harness's walk_forward loop with explicit (train, test) index splits.
    With splits = TimeSeriesSplit(n, gap).split(X) it reproduces
    evaluation_harness.walk_forward exactly (tested).
    """
    X = data[feature_columns]
    targets = {REGRESSION: data["future_return"].to_numpy(), CLASSIFICATION: data["direction"].to_numpy()}
    dates = data["Date"]
    pred_rows, fold_rows, folds = [], [], []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        folds.append({"fold": fold,
                      "train_start": dates.iloc[train_idx[0]].date().isoformat(),
                      "train_end": dates.iloc[train_idx[-1]].date().isoformat(),
                      "test_start": dates.iloc[test_idx[0]].date().isoformat(),
                      "test_end": dates.iloc[test_idx[-1]].date().isoformat(),
                      "n_train": int(len(train_idx)), "n_test": int(len(test_idx))})
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        for cand in candidates:
            y = targets[cand.task]
            estimator = clone(cand.factory())
            estimator.fit(X_train, y[train_idx])
            pred = predict_output(estimator, cand.task, X_test)
            pred_rows.append(pd.DataFrame({
                "row": test_idx, "Date": dates.iloc[test_idx].dt.date.astype(str).to_numpy(), "fold": fold,
                "model": cand.name, "task": cand.task, "y_true": y[test_idx], "prediction": pred}))
            metric_fn = regression_metrics if cand.task == REGRESSION else classification_metrics
            fold_rows.append({"fold": fold, "model": cand.name, "task": cand.task, **metric_fn(y[test_idx], pred)})
    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(fold_rows), folds


def holdout_splits(dates: pd.Series, horizon: int, n_splits: int, start: date, end: date
                   ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous test folds over the labelled holdout rows; train = all rows before fold start - horizon."""
    d = dates.dt.date.to_numpy()
    test_all = np.flatnonzero((d >= start) & (d <= end))
    if len(test_all) < n_splits:
        raise ResearchError(f"only {len(test_all)} labelled holdout rows for {n_splits} folds")
    if not np.array_equal(test_all, np.arange(test_all[0], test_all[-1] + 1)):
        raise ResearchError("holdout rows are not contiguous")
    splits = []
    for test_idx in np.array_split(test_all, n_splits):
        train_end = int(test_idx[0]) - horizon
        if train_end < 1:
            raise ResearchError("no training rows before the holdout")
        splits.append((np.arange(0, train_end), test_idx))
    return splits


def evaluate_on_holdout(frame: pd.DataFrame, feature_columns: list[str], candidates: list[Candidate], *,
                        horizon: int, n_splits: int, start: date, end: date) -> dict:
    data = frame[["Date", "Close", *feature_columns]].copy()
    validate_dataset(data, feature_columns)
    evaluable = build_targets(data, horizon)
    splits = holdout_splits(evaluable["Date"], horizon, n_splits, start, end)
    predictions, fold_metrics, folds = walk_forward_with_splits(evaluable, feature_columns, candidates, splits)
    counts = predictions.groupby("model")["row"].agg(["count", "nunique"])
    if not (counts["count"] == counts["nunique"]).all():
        raise RuntimeError("A holdout row was predicted more than once for the same model")
    config = HarnessConfig(horizon=horizon, n_splits=n_splits)
    metrics = pooled_metrics(predictions, candidates)
    gate = qualify(predictions, candidates, metrics, config)
    return {"metrics": metrics, "gate": gate, "folds": folds, "fold_metrics": fold_metrics.to_dict(orient="records"),
            "evaluation_period": {"start": folds[0]["test_start"], "end": folds[-1]["test_end"],
                                  "n_oos_rows": int(sum(f["n_test"] for f in folds))},
            "gap": config.gap, "n_splits": n_splits, "gate_version": GATE_VERSION}


def run_holdout(registry_path: Path, frame: pd.DataFrame | None, context_provenance: dict | None, *,
                confirm: bool, spec: ResearchSpec = OFFICIAL, results_dir: Path | None = None, log=print) -> dict:
    if not confirm:
        raise ResearchError("the confirmation holdout runs only with explicit authorization (--confirm)")
    registry_path = Path(registry_path)
    reg = load_registry(registry_path)
    if reg["status"] != "DEVELOPMENT_COMPLETE" or reg["holdout"] is not None:
        raise ResearchError("holdout requires a completed development stage and runs exactly once")
    matrix = _check_spec(reg, spec)
    qualifiers = list(reg["development"]["qualifiers"])
    now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: E731

    if not qualifiers:
        reg["holdout"] = {"status": "NOT_APPLICABLE", "result": NO_DEVELOPMENT_QUALIFIERS, "recorded_at": now(),
                          "explanation": ("No pre-registered configuration qualified under gate_v1 in development, "
                                          "so there is nothing to confirm; the holdout was not evaluated."),
                          "confirmed": []}
        reg["status"] = "COMPLETE"
        _write_json_atomic(registry_path, reg)
        return reg

    if frame is None:
        raise ResearchError("development qualifiers exist; the research frame is required")
    results_dir = Path(results_dir) if results_dir is not None else registry_path.parent
    by_name = {c.name: c for c in spec.candidates()}
    started, confirmed = now(), []
    for exp_id in qualifiers:
        entry = reg["results"][exp_id]
        cand = by_name[entry["model"]]
        cands = [c for c in spec.candidates() if c.task == cand.task and c.is_baseline] + [cand]
        cols = list(spec.feature_sets[entry["feature_set"]]["columns"])
        log(f"[holdout] {exp_id} ...")
        out = evaluate_on_holdout(frame, cols, cands, horizon=entry["horizon"], n_splits=spec.n_splits,
                                  start=spec.holdout_start, end=spec.holdout_end)
        artifact = _write_new_json(results_dir / "experiments" / f"holdout_{exp_id}.json",
                                   {"experiment_id": exp_id, "feature_columns": cols, **out})
        g = out["gate"][cand.name]
        is_confirmed = entry["development"]["status"] == QUALIFIED and g["status"] == QUALIFIED
        if is_confirmed:
            confirmed.append(exp_id)
        entry["holdout"] = {"oos_period": out["evaluation_period"], "n_oos_rows": out["evaluation_period"]["n_oos_rows"],
                            "n_splits": spec.n_splits, "gap": out["gap"], "metrics": out["metrics"][cand.name],
                            "baseline_metrics": {c.name: out["metrics"][c.name] for c in cands if c.is_baseline},
                            "gate": g, "status": g["status"], "p_value_raw": g["diebold_mariano"]["p_value"],
                            "artifact": artifact}
        entry["status"] = "CONFIRMED" if is_confirmed else "NOT_CONFIRMED"
    reg["holdout"] = {"status": "COMPLETE", "started_at": started, "finished_at": now(), "git": git_info(),
                      "environment": environment(), "context_inputs": context_provenance,
                      "evaluated": qualifiers, "confirmed": confirmed,
                      "result": (f"CONFIRMED: {confirmed}" if confirmed else NO_CONFIRMED_MODEL)}
    reg["status"] = "COMPLETE"
    _write_json_atomic(registry_path, reg)
    return reg


# ==========================================
# CLI
# ==========================================


def _print_status(reg: dict) -> None:
    print(f"Registry status : {reg['status']}   matrix sha256 {reg['matrix_sha256'][:16]}...")
    print(f"Experiments     : {len(reg['matrix']['experiments'])} pre-registered")
    dev = reg.get("development")
    if dev:
        mcmp = dev["multiple_comparisons"]
        print(f"Development     : {dev['frame']['first_date']} .. {dev['frame']['last_date']} "
              f"({dev['frame']['rows']} rows); {dev['n_tests']} tests; raw p<0.05: {mcmp['n_raw_p_below_alpha']}; "
              f"Holm p<0.05: {mcmp['n_holm_p_below_alpha']}")
        print(f"                  {dev['result']}  {dev['qualifiers'] or ''}")
    if reg.get("holdout"):
        print(f"Holdout         : {reg['holdout']['status']} - {reg['holdout']['result']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 9 predictive signal research (pre-registered)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "validate-context", "develop", "holdout", "status"):
        p = sub.add_parser(name)
        p.add_argument("--registry", type=Path, default=REGISTRY_PATH)
        p.add_argument("--store", type=Path, default=DEFAULT_STORE)
        if name in ("validate-context", "develop", "holdout"):
            p.add_argument("--spy", type=Path, required=True)
            p.add_argument("--qqq", type=Path, required=True)
        if name == "holdout":
            p.add_argument("--confirm", action="store_true", help="explicit authorization to run the holdout")
    args = parser.parse_args(argv)

    if args.command == "freeze":
        reg = freeze(args.registry, inputs=input_provenance(args.store))
        print(f"Frozen: {args.registry}")
        print(f"  matrix sha256 {reg['matrix_sha256']}")
        print(f"  {len(reg['matrix']['experiments'])} experiments = {len(reg['matrix']['feature_sets'])} feature sets "
              f"x {len(reg['matrix']['horizons'])} horizons x "
              f"{sum(not c['is_baseline'] for c in reg['matrix']['candidates'])} models")
        return 0
    if args.command == "status":
        _print_status(load_registry(args.registry))
        return 0

    if args.command == "holdout" and not args.confirm:
        raise ResearchError("the confirmation holdout runs only with explicit authorization (--confirm)")
    if args.command == "holdout":
        reg = load_registry(args.registry)
        if reg["status"] == "DEVELOPMENT_COMPLETE" and not reg["development"]["qualifiers"]:
            _print_status(run_holdout(args.registry, None, None, confirm=True))
            return 0

    frame, prov = build_research_frame(args.store, args.spy, args.qqq)
    if args.command == "validate-context":
        f = prov["frame"]
        print(f"Research frame : {f['rows']} rows, {f['first_date']} .. {f['last_date']} "
              f"({f['rows_before_full_sentiment_window']} early store rows lack a full 20-session window)")
        dev = development_frame(frame, OFFICIAL.dev_end)
        print(f"Development    : {len(dev)} rows, {dev['Date'].iloc[0].date()} .. {dev['Date'].iloc[-1].date()}")
        hold = frame[frame["Date"].dt.date >= OFFICIAL.holdout_start]
        print(f"Holdout rows   : {len(hold)}, {hold['Date'].iloc[0].date()} .. {hold['Date'].iloc[-1].date()}")
        for k in ("spy_snapshot", "qqq_snapshot"):
            s = prov[k]
            print(f"{s['ticker']:4s} snapshot  : {s['path']}  {s['first_date']} .. {s['last_date']}  "
                  f"sha256 {s['sha256'][:16]}...  retrieved {s['retrieved_at_utc']}")
        print(f"Features       : {f['n_feature_columns']} columns, no NaN/inf; SPY/QQQ aligned to every AAPL session")
        for c in MARKET_CONTEXT_COLUMNS + SENTIMENT_WINDOW_COLUMNS:
            v = frame[c]
            print(f"  {c:26s} mean {v.mean(): .6f}  std {v.std(): .6f}  min {v.min(): .6f}  max {v.max(): .6f}")
        return 0
    if args.command == "develop":
        _print_status(run_development(args.registry, frame, prov))
        return 0
    _print_status(run_holdout(args.registry, frame, prov, confirm=args.confirm))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
