"""
Phase 10A - excess-return signal research (AAPL relative to SPY), pre-registered.

Question: can the existing leakage-safe technical / sentiment / event /
market-context features predict AAPL's future return RELATIVE TO SPY?
A valid outcome is NO QUALIFIED MODEL.

    freeze              persist the complete matrix (feature sets, horizons,
                        targets, models + fixed parameters, gate, dates, F
                        duplicate resolution) and the verified input hashes,
                        plus structural freeze validation - no model is fitted
    target-diagnostics  AFTER freeze: target distributions on DEVELOPMENT rows
                        only; holdout gets structural row/label counts only
    develop             run exactly the frozen 198 experiments on sessions
                        <= 2024-09-30; no qualifier -> NO DEVELOPMENT QUALIFIERS
                        and the holdout is recorded NOT_APPLICABLE (never read)
    holdout --confirm   only with development qualifiers and explicit
                        authorization; each qualifier exactly once
    status

Harness adaptation (the only one): the central harness builds targets from a
single Close series (evaluation_harness.build_targets), which cannot express
a difference of two returns. Phase 10A builds the targets with
training/excess_targets.build_excess_targets and then uses the harness's own
validate_dataset, walk_forward (TimeSeriesSplit(n_splits, gap = horizon)),
pooled_metrics and qualify (gate_v1) unchanged. With SPY held constant the
pipeline reproduces run_evaluation exactly (tested).

Baselines keep their harness names because qualify() finds the
classification references by name ("Base Rate", "Always UP"); reports label
them Mean / Zero Excess Return and Excess Direction Base Rate.

Usage (project root):
    python -m training.phase10a_research freeze
    python -m training.phase10a_research target-diagnostics
    python -m training.phase10a_research develop
    python -m training.phase10a_research holdout --confirm
    python -m training.phase10a_research status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services import market_context as mc  # noqa: E402
from training.build_dataset import file_sha256, load_raw_snapshot  # noqa: E402
from training.evaluation_harness import (  # noqa: E402
    CLASSIFICATION,
    GATE_VERSION,
    HARNESS_VERSION,
    NO_QUALIFIED_MODEL,
    QUALIFIED,
    REGRESSION,
    Candidate,
    HarnessConfig,
    git_info,
    pooled_metrics,
    qualify,
    validate_dataset,
    walk_forward,
)
from training.excess_targets import (  # noqa: E402
    HORIZONS,
    SPY_CLOSE,
    TARGET_COLUMNS,
    build_excess_targets,
    target_definition,
)
from training.feature_store import EVENT_NUMERIC_COLUMNS, SENTIMENT_COLUMNS, TECHNICAL_COLUMNS  # noqa: E402
from training.phase9_research import (  # noqa: E402
    DEFAULT_STORE,
    MARKET_CONTEXT_COLUMNS,
    SENTIMENT_WINDOW_COLUMNS,
    ResearchError,
    _rel,
    _slug,
    _write_json_atomic,
    _write_new_json,
    build_research_frame,
    candidate_spec,
    canonical_json,
    environment,
    holdout_splits,
    holm_adjust,
    input_provenance,
    phase9_candidates,
    walk_forward_with_splits,
)
from training.targets import target_summary  # noqa: E402

PHASE = "10A"
MATRIX_VERSION = "phase10a_matrix_v1"
REGISTRY_VERSION = "phase10a_registry_v1"
SYMBOL = "AAPL"
N_SPLITS = 20
DEV_START = date(2017, 2, 1)
DEV_END = date(2024, 9, 30)
HOLDOUT_START = date(2024, 10, 1)
HOLDOUT_END = date(2026, 9, 25)

RESEARCH_DIR = PROJECT_ROOT / "data" / "results" / "research" / "phase10a"
REGISTRY_PATH = RESEARCH_DIR / "registry.json"
DEFAULT_SPY = PROJECT_ROOT / "data" / "raw" / "stocks" / "SPY_1d_10y_20261001T104943Z.csv"
DEFAULT_QQQ = PROJECT_ROOT / "data" / "raw" / "stocks" / "QQQ_1d_10y_20261001T104948Z.csv"

NO_DEVELOPMENT_QUALIFIERS = "NO DEVELOPMENT QUALIFIERS"
NO_CONFIRMED_MODEL = "NO CONFIRMED MODEL"
DETERMINISM_NOT_VERIFIED = "DEVELOPMENT RERUN DETERMINISM NOT VERIFIED"

# ---------- F duplicate resolution (representation hygiene, fixed before any result) ----------
DUPLICATE_RESOLUTION = {
    "duplicate": ["event__article_count", "sw_count_1"],
    "dropped": "sw_count_1",
    "retained": "event__article_count",
    "reason": ("both count AAPL articles with ts(prev session) < information_available_at <= ts(D) "
               "(identical on every Phase 9 research row); the retained column is the verified "
               "Phase 6/8 event feature, so the events family is identical in C, D and F; "
               "sw_count_surprise_1_20 keeps using the same 1-session count"),
    "nature": "representation hygiene, decided before any Phase 10A result; Phase 9 artifacts unchanged",
}
SENTIMENT_WINDOW_10A = [c for c in SENTIMENT_WINDOW_COLUMNS if c != DUPLICATE_RESOLUTION["dropped"]]

FEATURE_SETS = {
    "A": {"families": ["technical"], "columns": list(TECHNICAL_COLUMNS)},
    "B": {"families": ["technical", "sentiment"], "columns": [*TECHNICAL_COLUMNS, *SENTIMENT_COLUMNS]},
    "C": {"families": ["technical", "events"], "columns": [*TECHNICAL_COLUMNS, *EVENT_NUMERIC_COLUMNS]},
    "D": {"families": ["technical", "sentiment", "events"],
          "columns": [*TECHNICAL_COLUMNS, *SENTIMENT_COLUMNS, *EVENT_NUMERIC_COLUMNS]},
    "E": {"families": ["technical", "market_context"], "columns": [*TECHNICAL_COLUMNS, *MARKET_CONTEXT_COLUMNS]},
    "F": {"families": ["technical", "sentiment_window (without sw_count_1)", "events", "market_context"],
          "columns": [*TECHNICAL_COLUMNS, *SENTIMENT_WINDOW_10A, *EVENT_NUMERIC_COLUMNS, *MARKET_CONTEXT_COLUMNS]},
}
ALL_FEATURE_COLUMNS = list(dict.fromkeys(c for fs in FEATURE_SETS.values() for c in fs["columns"]))

BASELINE_LABELS = {
    "Mean Return": "Mean Excess Return - training-fold mean of future_excess_return_h",
    "Zero Return": "Zero Excess Return - constant 0 (AAPL matches SPY)",
    "Base Rate": "Excess Direction Base Rate - P(UP) = training-fold share of excess_direction_h == 1",
    "Always UP": "Always UP - always predicts excess_direction_h = 1 with probability 1",
}


class Phase10AError(ResearchError):
    """The Phase 10A protocol would be violated."""


def experiment_id(feature_set: str, horizon: int, model: str) -> str:
    return f"p10a_{feature_set}_h{horizon}_{_slug(model)}"


@dataclass(frozen=True)
class Spec10A:
    """Everything that defines the Phase 10A matrix. OFFICIAL is the pre-registration."""

    feature_sets: dict = field(default_factory=lambda: FEATURE_SETS)
    horizons: tuple = HORIZONS
    candidates_factory: Callable[[], list[Candidate]] = phase9_candidates
    n_splits: int = N_SPLITS
    dev_start: date = DEV_START
    dev_end: date = DEV_END
    holdout_start: date = HOLDOUT_START
    holdout_end: date = HOLDOUT_END

    def candidates(self) -> list[Candidate]:
        cands = self.candidates_factory()
        names = [c.name for c in cands]
        if len(set(names)) != len(names):
            raise Phase10AError(f"candidate names must be unique: {names}")
        return cands

    def matrix(self) -> dict:
        cands = self.candidates()
        for fs, d in self.feature_sets.items():
            cols = list(d["columns"])
            if len(set(cols)) != len(cols):
                raise Phase10AError(f"feature set {fs} lists a column twice")
            leaked = sorted(set(cols) & set(TARGET_COLUMNS))
            if leaked:
                raise Phase10AError(f"feature set {fs} contains target/price columns {leaked}")
        experiments = [{"experiment_id": experiment_id(fs, h, c.name), "feature_set": fs, "horizon": h,
                        "model": c.name, "task": c.task}
                       for fs in self.feature_sets for h in self.horizons for c in cands if not c.is_baseline]
        return {
            "phase": PHASE,
            "matrix_version": MATRIX_VERSION,
            "symbol": SYMBOL,
            "benchmark": "SPY",
            "research_question": ("Can leakage-safe technical, sentiment, event and market-context features "
                                  "predict AAPL's future excess return relative to SPY?"),
            "hypothesis": ("these features may carry significant out-of-sample information about AAPL "
                           "relative to SPY even without it for AAPL's absolute return (to be tested, not assumed)"),
            "feature_sets": {k: {"families": list(v["families"]), "columns": list(v["columns"]),
                                 "n_features": len(v["columns"])} for k, v in self.feature_sets.items()},
            "f_duplicate_resolution": DUPLICATE_RESOLUTION,
            "feature_definitions": {
                "technical": "technical_v2 (unchanged)",
                "sentiment": "sentiment_features_v1 (unchanged, cumulative), prefix sentiment__",
                "events": "event_features_v1 numeric (unchanged), prefix event__",
                "sentiment_window": "sentiment_window_v1 (Phase 9, unchanged) without sw_count_1 in F",
                "market_context": "market_context_v1 (Phase 9, unchanged)",
            },
            "horizons": list(self.horizons),
            "targets": {str(h): target_definition(h) for h in self.horizons},
            "candidates": [{**candidate_spec(c), "report_label": BASELINE_LABELS.get(c.name, c.name)} for c in cands],
            "evaluation": {
                "harness_components": ("evaluation_harness.validate_dataset, walk_forward, pooled_metrics, "
                                       "qualify (unchanged)"),
                "target_construction": ("training.excess_targets.build_excess_targets replaces "
                                        "evaluation_harness.build_targets (difference of two returns)"),
                "harness_version": HARNESS_VERSION,
                "gate_version": GATE_VERSION,
                "gate": {
                    "regression": "MSE < min(MSE of Mean Return, Zero Return) AND one-sided DM p < significance",
                    "classification": ("log loss < Base Rate log loss AND one-sided DM p < significance "
                                       "AND accuracy >= Always UP accuracy"),
                    "dm": "one-sided Diebold-Mariano, Newey-West variance with horizon - 1 lags",
                },
                "significance_level": HarnessConfig().significance_level,
                "validation": "TimeSeriesSplit walk-forward, pooled out-of-sample",
                "n_splits": self.n_splits,
                "gap": "horizon",
                "baselines": {"regression": ["Mean Return", "Zero Return"],
                              "classification": ["Always UP", "Base Rate"],
                              "report_labels": BASELINE_LABELS},
                "tuning": "none - every parameter fixed in this matrix",
                "multiple_comparisons": {"method": "Holm", "scope": "all 198 development tests",
                                         "role": "informational only; gate_v1 is authoritative"},
                "row_selection": ("Phase 9 research frame (Phase 8 store rows covered by every family incl. the "
                                  "20-session sentiment window and strict market context); identical rows for "
                                  "every feature set"),
                "development_period": {"start": self.dev_start.isoformat(), "end": self.dev_end.isoformat(),
                                       "rule": ("frame cut at the end BEFORE targets are built: the last h rows "
                                                "are unlabelled and no AAPL/SPY close after the end is used")},
                "confirmation_holdout": {"name": "Phase-10A confirmation holdout",
                                         "start": self.holdout_start.isoformat(),
                                         "end": self.holdout_end.isoformat(),
                                         "limitation": "not pristine: earlier phases evaluated parts of it",
                                         "rule": ("evaluated only for development qualifiers, only with --confirm, "
                                                  "each exactly once; with no qualifier it is never read")},
                "holdout_procedure": (f"labelled holdout rows in {self.n_splits} contiguous folds; expanding-window "
                                      "training on all earlier rows minus a gap of `horizon` rows; same metrics "
                                      "and gate_v1 (Phase 9 procedure)"),
                "confirmation_rule": "CONFIRMED iff QUALIFIED in development AND on the holdout",
            },
            "leakage_rules": [
                "prediction timestamp D 16:30 America/New_York; features use information_available_at <= it",
                "market features: AAPL/SPY/QQQ bars dated <= D only",
                "news/sentiment/events: verified Phase 8 records with the established conservative availability",
                "future AAPL and SPY closes: target construction only",
                "no forward fill, interpolation or normalization with future values",
                "development frame cut at the development end before targets are built",
                "no confirmation-period target distribution is inspected",
            ],
            "experiments": experiments,
        }


OFFICIAL = Spec10A()


def matrix_sha256(matrix: dict) -> str:
    return hashlib.sha256(canonical_json(matrix).encode("utf-8")).hexdigest()


# ==========================================
# Inputs and research frame
# ==========================================


def collect_inputs(store_path: Path = DEFAULT_STORE, spy_path: Path = DEFAULT_SPY, qqq_path: Path = DEFAULT_QQQ) -> dict:
    """Verified Phase 8/9 input chain + the SPY/QQQ snapshots (hash checked against their metadata)."""
    inputs = input_provenance(store_path)
    for key, path, ticker in (("spy_snapshot", spy_path, "SPY"), ("qqq_snapshot", qqq_path, "QQQ")):
        _, meta = mc.load_context_snapshot(path, ticker)
        inputs[key] = {"path": _rel(path), "sha256": meta["sha256"]}
    return inputs


def build_frame(store_path: Path = DEFAULT_STORE, spy_path: Path = DEFAULT_SPY,
                qqq_path: Path = DEFAULT_QQQ) -> tuple[pd.DataFrame, dict]:
    """Phase 9 research frame (read-only) + SPY_Close from the verified SPY snapshot (targets only)."""
    frame, prov = build_research_frame(store_path, spy_path, qqq_path)
    spy, _ = mc.load_context_snapshot(spy_path, "SPY")
    spy_close = dict(zip(spy["Date"].dt.date, spy["Close"].astype("float64")))
    dates = [ts.date() for ts in frame["Date"]]
    missing = [d for d in dates if d not in spy_close]
    if missing:
        raise Phase10AError(f"SPY close missing for {len(missing)} research row(s), e.g. {missing[:3]}")
    frame = frame.copy()
    frame[SPY_CLOSE] = [spy_close[d] for d in dates]
    return frame, prov


def frame_sha256(frame: pd.DataFrame) -> str:
    cols = ["Date", "Close", SPY_CLOSE, *ALL_FEATURE_COLUMNS]
    return hashlib.sha256(frame[cols].to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def development_frame(frame: pd.DataFrame, spec: Spec10A = OFFICIAL) -> pd.DataFrame:
    d = frame["Date"].dt.date
    dev = frame[(d >= spec.dev_start) & (d <= spec.dev_end)].reset_index(drop=True)
    if dev.empty or dev["Date"].iloc[-1].date() > spec.dev_end:
        raise Phase10AError("development frame is empty or extends past the development end")
    return dev


def holdout_label_counts(frame: pd.DataFrame, horizon: int, spec: Spec10A = OFFICIAL) -> dict:
    """STRUCTURAL ONLY: how many holdout rows have an h-day label (no target value is computed)."""
    d = frame["Date"].dt.date.to_numpy()
    idx = np.flatnonzero((d >= spec.holdout_start) & (d <= spec.holdout_end))
    labelled = idx[idx + horizon < len(frame)]
    return {"holdout_rows": int(len(idx)), "labelled_rows": int(len(labelled)),
            "first_date": d[labelled[0]].isoformat() if len(labelled) else None,
            "last_labelled_date": d[labelled[-1]].isoformat() if len(labelled) else None}


def freeze_validation(frame: pd.DataFrame, prov: dict, aapl_bars: pd.DataFrame, spec: Spec10A = OFFICIAL,
                      rebuilt_frame: pd.DataFrame | None = None) -> dict:
    """Structural checks before freezing. Fits nothing; computes no development distribution."""
    a, b = DUPLICATE_RESOLUTION["duplicate"]
    if not np.array_equal(frame[a].to_numpy(dtype="float64"), frame[b].to_numpy(dtype="float64")):
        raise Phase10AError(f"{a} and {b} are not identical on every research row; refusing to drop {b}")
    for fs, dfn in spec.feature_sets.items():
        missing = [c for c in dfn["columns"] if c not in frame.columns]
        if missing:
            raise Phase10AError(f"feature set {fs}: columns missing from the research frame: {missing}")
    snap = dict(zip(aapl_bars["Date"].dt.date, aapl_bars["Close"].astype("float64")))
    store_close = frame["Close"].to_numpy(dtype="float64")
    snap_close = np.array([snap[ts.date()] for ts in frame["Date"]])
    rel = float(np.max(np.abs(store_close / snap_close - 1)))
    if rel > 1e-12:
        raise Phase10AError(f"store Close differs from the AAPL snapshot Close (max relative diff {rel})")
    dev = development_frame(frame, spec)
    dupe_pairs = {fs: [(x, y) for i, x in enumerate(d["columns"]) for y in d["columns"][i + 1:]
                       if frame[x].equals(frame[y])] for fs, d in spec.feature_sets.items()}
    if any(dupe_pairs.values()):
        raise Phase10AError(f"exact duplicate columns inside a feature set: {dupe_pairs}")
    out = {
        "frame": {**prov["frame"], "rows_with_spy_close": len(frame)},
        "development": {"rows": len(dev), "first_date": dev["Date"].iloc[0].date().isoformat(),
                        "last_date": dev["Date"].iloc[-1].date().isoformat()},
        "exclusions": {"store_rows_without_full_20_session_window": prov["frame"]["rows_before_full_sentiment_window"],
                       "unlabelled_rows_per_horizon": "last h development rows (no close after the development end)"},
        "duplicate_check": {"columns": [a, b], "rows_compared": len(frame), "identical": True,
                            "dropped": DUPLICATE_RESOLUTION["dropped"]},
        "no_duplicate_columns_within_feature_sets": True,
        "store_close_equals_aapl_snapshot_close": {"max_relative_difference": rel},
        "feature_availability": {fs: {"n_features": len(d["columns"]),
                                      "complete_rows": int(np.isfinite(frame[d["columns"]].to_numpy(dtype="float64"))
                                                           .all(axis=1).sum())}
                                 for fs, d in spec.feature_sets.items()},
        "target_availability": {str(h): {"development_labelled_rows": len(dev) - h,
                                         "holdout": holdout_label_counts(frame, h, spec)} for h in spec.horizons},
        "leakage": {"target_columns_in_any_feature_set": sorted(set(ALL_FEATURE_COLUMNS) & set(TARGET_COLUMNS))},
    }
    # construction determinism (development targets only; holdout targets are never computed here)
    m1, m2 = spec.matrix(), spec.matrix()
    det = {"matrix_identical": matrix_sha256(m1) == matrix_sha256(m2)}
    if rebuilt_frame is not None:
        det["frame_sha256"] = frame_sha256(frame)
        det["frame_rebuild_identical"] = frame_sha256(frame) == frame_sha256(rebuilt_frame)
    tgt = {}
    for h in spec.horizons:
        t1 = build_excess_targets(dev[["Date", "Close", SPY_CLOSE]], h)
        t2 = build_excess_targets(dev[["Date", "Close", SPY_CLOSE]], h)
        h1 = hashlib.sha256(t1.to_csv(index=False).encode()).hexdigest()
        tgt[str(h)] = h1 == hashlib.sha256(t2.to_csv(index=False).encode()).hexdigest()
    det["development_targets_identical"] = tgt
    det["full_development_rerun"] = DETERMINISM_NOT_VERIFIED
    out["construction_determinism"] = det
    return out


# ==========================================
# Registry
# ==========================================


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    path = Path(path)
    if not path.is_file():
        raise Phase10AError(f"registry not found: {path} (run `freeze` first)")
    reg = json.loads(path.read_text(encoding="utf-8"))
    if reg.get("registry_version") != REGISTRY_VERSION or reg.get("phase") != PHASE:
        raise Phase10AError("not a Phase 10A registry")
    if matrix_sha256(reg["matrix"]) != reg["matrix_sha256"]:
        raise Phase10AError("registry matrix does not match its recorded hash (edited after freezing)")
    return reg


def _check_spec(reg: dict, spec: Spec10A) -> dict:
    matrix = spec.matrix()
    if matrix_sha256(matrix) != reg["matrix_sha256"]:
        raise Phase10AError("the experiment matrix in code differs from the FROZEN matrix")
    return matrix


def _check_inputs(reg: dict, inputs: dict) -> None:
    if inputs != reg["inputs"]:
        changed = sorted(k for k in set(inputs) | set(reg["inputs"]) if inputs.get(k) != reg["inputs"].get(k))
        raise Phase10AError(f"input hashes differ from the frozen registry: {changed} - stopping (nothing regenerated)")


def freeze(registry_path: Path = REGISTRY_PATH, *, inputs: dict, validation: dict,
           spec: Spec10A = OFFICIAL) -> dict:
    registry_path = Path(registry_path)
    if registry_path.exists():
        raise Phase10AError(f"{registry_path} already exists; the Phase 10A matrix is frozen once")
    matrix = spec.matrix()
    reg = {
        "registry_version": REGISTRY_VERSION, "phase": PHASE,
        "matrix": matrix, "matrix_sha256": matrix_sha256(matrix),
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_git": git_info(), "frozen_environment": environment(),
        "inputs": inputs, "freeze_validation": validation,
        "status": "FROZEN", "target_diagnostics": None, "development": None, "holdout": None,
        "determinism": DETERMINISM_NOT_VERIFIED,
        "results": {e["experiment_id"]: {**e, "status": "PENDING", "development": None, "holdout": None}
                    for e in matrix["experiments"]},
    }
    _write_json_atomic(registry_path, reg)
    return reg


# ==========================================
# Target diagnostics (development only)
# ==========================================


def _stats(values: pd.Series) -> dict:
    v = values.to_numpy(dtype="float64")
    q = np.quantile(v, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {"count": int(len(v)), "finite": int(np.isfinite(v).sum()), "missing": int(np.isnan(v).sum()),
            "mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std(ddof=1)),
            "min": float(v.min()), "max": float(v.max()),
            "quantiles": dict(zip(["p01", "p05", "p25", "p50", "p75", "p95", "p99"], map(float, q))),
            "skew": float(pd.Series(v).skew())}


def target_diagnostics(frame: pd.DataFrame, spec: Spec10A = OFFICIAL) -> dict:
    dev = development_frame(frame, spec)
    out = {"scope": "development rows only (<= development end); holdout: structural counts only",
           "development_rows": len(dev), "horizons": {}}
    close, spy = dev["Close"].to_numpy(), dev[SPY_CLOSE].to_numpy()
    for h in spec.horizons:
        t = build_excess_targets(dev[["Date", "Close", SPY_CLOSE]], h)
        n = len(t)
        picks = sorted({0, 1, n // 4, n // 2, (3 * n) // 4, n - 2, n - 1})
        spot = []
        for i in picks:                                   # independent recomputation from raw closes
            a = close[i + h] / close[i] - 1
            s = spy[i + h] / spy[i] - 1
            spot.append({"row": int(i), "date": t["Date"].iloc[i].date().isoformat(),
                         "label_close_date": dev["Date"].iloc[i + h].date().isoformat(),
                         "aapl_leg": float(a), "spy_leg": float(s), "excess": float(t["future_return"].iloc[i]),
                         "matches": bool(abs((a - s) - t["future_return"].iloc[i]) <= 1e-15
                                         and t["direction"].iloc[i] == int((a - s) > 0))})
        out["horizons"][str(h)] = {
            "target": target_definition(h)["regression"]["name"],
            "labelled_rows": n, "unlabelled_tail_rows": h,
            "last_label_close_date": dev["Date"].iloc[n - 1 + h].date().isoformat(),
            "future_excess_return": _stats(t["future_return"]),
            "aapl_leg": _stats(t["aapl_future_return"]), "spy_leg": _stats(t["spy_future_return"]),
            "excess_direction": {**target_summary(t["future_return"], t["direction"]),
                                 "positive_pct": float(100 * t["direction"].mean()),
                                 "non_positive_pct": float(100 * (1 - t["direction"].mean()))},
            "spot_checks": spot, "all_spot_checks_match": all(s["matches"] for s in spot),
            "holdout_structural": holdout_label_counts(frame, h, spec),
        }
    out["target_columns_in_any_feature_set"] = sorted(set(ALL_FEATURE_COLUMNS) & set(TARGET_COLUMNS))
    return out


def run_target_diagnostics(registry_path: Path, frame: pd.DataFrame, inputs: dict, *, spec: Spec10A = OFFICIAL,
                           results_dir: Path | None = None) -> dict:
    registry_path = Path(registry_path)
    reg = load_registry(registry_path)
    _check_spec(reg, spec)
    _check_inputs(reg, inputs)
    if reg["target_diagnostics"] is not None or reg["development"] is not None:
        raise Phase10AError("target diagnostics already recorded (or development already run)")
    results_dir = Path(results_dir) if results_dir is not None else registry_path.parent
    diag = target_diagnostics(frame, spec)
    artifact = _write_new_json(results_dir / "targets" / "target_diagnostics_development.json", diag)
    reg["target_diagnostics"] = {"artifact": artifact, "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _write_json_atomic(registry_path, reg)
    return diag


# ==========================================
# Evaluation (harness components, excess targets)
# ==========================================


def evaluate_excess(frame: pd.DataFrame, feature_columns: list[str], candidates: list[Candidate], *,
                    horizon: int, n_splits: int, splits=None) -> dict:
    """
    validate_dataset -> build_excess_targets -> walk_forward (TimeSeriesSplit(n_splits, gap = h), or the
    given explicit splits for the holdout) -> pooled_metrics -> qualify. Warnings are recorded, not suppressed
    from the computation.
    """
    data = frame[["Date", "Close", SPY_CLOSE, *feature_columns]].copy()
    validate_dataset(data, feature_columns)
    evaluable = build_excess_targets(data, horizon)
    config = HarnessConfig(horizon=horizon, n_splits=n_splits)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if splits is None:
            predictions, fold_metrics, folds = walk_forward(evaluable, feature_columns, candidates, config)
        else:
            predictions, fold_metrics, folds = walk_forward_with_splits(
                evaluable, feature_columns, candidates, splits(evaluable))
    counts = predictions.groupby("model")["row"].agg(["count", "nunique"])
    if not (counts["count"] == counts["nunique"]).all():
        raise RuntimeError("A row was predicted more than once for the same model")
    metrics = pooled_metrics(predictions, candidates)
    gate = qualify(predictions, candidates, metrics, config)
    oos = np.sort(predictions["row"].unique())
    evaluated = [t for t in (REGRESSION, CLASSIFICATION) if any(c.task == t and not c.is_baseline for c in candidates)]
    qualified = {t: sorted(n for n, g in gate.items() if g["task"] == t and g["status"] == QUALIFIED) for t in evaluated}
    warn_counts = Counter(f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught)
    return {
        "harness_version": HARNESS_VERSION, "gate_version": GATE_VERSION,
        "significance_level": config.significance_level, "horizon_days": horizon,
        "n_splits": n_splits, "gap": config.gap, "features": list(feature_columns),
        "target": target_definition(horizon),
        "frame": {"rows": len(data), "first_date": data["Date"].iloc[0].date().isoformat(),
                  "last_date": data["Date"].iloc[-1].date().isoformat()},
        "target_summary": {"labelled_rows": target_summary(evaluable["future_return"], evaluable["direction"]),
                           "pooled_oos_rows": target_summary(evaluable["future_return"].iloc[oos],
                                                             evaluable["direction"].iloc[oos])},
        "training_period": {"start": folds[0]["train_start"], "end": folds[-1]["train_end"]},
        "evaluation_period": {"start": folds[0]["test_start"], "end": folds[-1]["test_end"],
                              "n_oos_rows": int(sum(f["n_test"] for f in folds))},
        "qualification_result": {t: (n if n else NO_QUALIFIED_MODEL) for t, n in qualified.items()},
        "models": [{"model": c.name, "report_label": BASELINE_LABELS.get(c.name, c.name), "task": c.task,
                    "is_baseline": c.is_baseline, "description": c.description,
                    "metrics": metrics[c.name], "qualification": gate[c.name]} for c in candidates],
        "baseline_metrics": {c.name: metrics[c.name] for c in candidates if c.is_baseline},
        "folds": folds, "fold_metrics": fold_metrics.to_dict(orient="records"),
        "warnings": {"total": len(caught), "by_message": dict(warn_counts)},
    }


# ==========================================
# Development
# ==========================================


def run_development(registry_path: Path, frame: pd.DataFrame, inputs: dict, *, spec: Spec10A = OFFICIAL,
                    results_dir: Path | None = None, evaluate=evaluate_excess, log=print) -> dict:
    registry_path = Path(registry_path)
    reg = load_registry(registry_path)
    if reg["status"] != "FROZEN" or reg["development"] is not None:
        raise Phase10AError("development has already been run for this registry")
    matrix = _check_spec(reg, spec)
    _check_inputs(reg, inputs)
    results_dir = Path(results_dir) if results_dir is not None else registry_path.parent
    exp_dir = results_dir / "experiments"
    planned = [exp_dir / f"dev_{fs}_h{h}.json" for fs in spec.feature_sets for h in spec.horizons]
    if any(p.exists() for p in planned):
        raise Phase10AError(f"development artifacts already exist in {exp_dir}")

    dev = development_frame(frame, spec)
    if dev["Date"].iloc[-1].date() > spec.dev_end:
        raise Phase10AError("development rows after the development end")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    specs = {c["name"]: c for c in matrix["candidates"]}
    entries, runs = [], []
    for fs, fdef in spec.feature_sets.items():
        cols = list(fdef["columns"])
        for h in spec.horizons:
            name = f"phase10a_dev_{fs}_h{h}"
            log(f"[develop] {name}: {len(cols)} features, horizon {h} ...")
            report = {"experiment": name, "phase": PHASE, "feature_set": fs, "matrix_sha256": reg["matrix_sha256"],
                      "inputs": inputs, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      **evaluate(dev, cols, spec.candidates(), horizon=h, n_splits=spec.n_splits)}
            artifact = _write_new_json(exp_dir / f"dev_{fs}_h{h}.json", report)
            models = {m["model"]: m for m in report["models"]}
            baselines = {t: {n: m["metrics"] for n, m in models.items() if m["is_baseline"] and m["task"] == t}
                         for t in (REGRESSION, CLASSIFICATION)}
            runs.append({"feature_set": fs, "horizon": h, "artifact": artifact,
                         "evaluation_period": report["evaluation_period"], "training_period": report["training_period"],
                         "qualification_result": report["qualification_result"], "baseline_metrics": baselines,
                         "warnings_total": report["warnings"]["total"]})
            for mname, m in models.items():
                if m["is_baseline"]:
                    continue
                q = m["qualification"]
                entries.append({
                    "experiment_id": experiment_id(fs, h, mname), "feature_set": fs, "horizon": h,
                    "model": mname, "task": m["task"], "target": matrix["targets"][str(h)][m["task"]],
                    "parameters": specs[mname]["params"], "random_seed": specs[mname]["random_seed"],
                    "development": {
                        "frame_period": [report["frame"]["first_date"], report["frame"]["last_date"]],
                        "frame_rows": report["frame"]["rows"],
                        "oos_period": report["evaluation_period"], "training_period": report["training_period"],
                        "n_oos_rows": report["evaluation_period"]["n_oos_rows"],
                        "n_splits": report["n_splits"], "gap": report["gap"],
                        "metrics": m["metrics"], "baseline_metrics": baselines[m["task"]],
                        "gate": q, "status": q["status"],
                        "p_value_raw": q["diebold_mariano"]["p_value"], "p_value_holm": None,
                        "artifact": artifact,
                    },
                })

    planned_ids = sorted(e["experiment_id"] for e in matrix["experiments"])
    if sorted(e["experiment_id"] for e in entries) != planned_ids:
        raise Phase10AError("development results do not cover exactly the pre-registered experiments")
    holm = holm_adjust([e["development"]["p_value_raw"] for e in entries])
    for e, p in zip(entries, holm):
        e["development"]["p_value_holm"] = p
    qualifiers = [e["experiment_id"] for e in entries if e["development"]["status"] == QUALIFIED]
    for e in entries:
        reg["results"][e["experiment_id"]].update({"status": "DEVELOPMENT_" + e["development"]["status"], **e})
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    reg["development"] = {
        "status": "COMPLETE", "started_at": started, "finished_at": now, "git": git_info(),
        "environment": environment(), "inputs_verified": True,
        "frame": {"rows": len(dev), "first_date": dev["Date"].iloc[0].date().isoformat(),
                  "last_date": dev["Date"].iloc[-1].date().isoformat()},
        "runs": runs, "n_tests": len(entries),
        "multiple_comparisons": {"method": "Holm", "n_tests": len(entries),
                                 "n_raw_p_below_alpha": sum(e["development"]["p_value_raw"] < 0.05 for e in entries),
                                 "n_holm_p_below_alpha": sum(p < 0.05 for p in holm),
                                 "role": "informational; gate_v1 decides qualification"},
        "warnings_total": sum(r["warnings_total"] for r in runs),
        "qualifiers": qualifiers,
        "result": (f"{len(qualifiers)} DEVELOPMENT QUALIFIER(S)" if qualifiers else NO_DEVELOPMENT_QUALIFIERS),
        "determinism": DETERMINISM_NOT_VERIFIED,
    }
    if qualifiers:
        reg["status"] = "DEVELOPMENT_COMPLETE"            # holdout needs explicit authorization
    else:                                                   # nothing to confirm: the holdout is never read
        reg["holdout"] = {"status": "NOT_APPLICABLE", "result": NO_DEVELOPMENT_QUALIFIERS, "recorded_at": now,
                          "explanation": "no development qualifier; the Phase-10A confirmation holdout was not read",
                          "confirmed": []}
        reg["status"] = "COMPLETE"
    _write_json_atomic(registry_path, reg)
    return reg


# ==========================================
# Confirmation holdout (qualifiers + explicit authorization only)
# ==========================================


def run_holdout(registry_path: Path, frame: pd.DataFrame | None, inputs: dict | None, *, confirm: bool,
                spec: Spec10A = OFFICIAL, results_dir: Path | None = None, log=print) -> dict:
    if not confirm:
        raise Phase10AError("the Phase-10A confirmation holdout runs only with explicit authorization (--confirm)")
    registry_path = Path(registry_path)
    reg = load_registry(registry_path)
    if reg["status"] != "DEVELOPMENT_COMPLETE" or reg["holdout"] is not None:
        raise Phase10AError("holdout requires development qualifiers and runs exactly once "
                            f"(registry status {reg['status']})")
    _check_spec(reg, spec)
    qualifiers = list(reg["development"]["qualifiers"])
    if not qualifiers:
        raise Phase10AError("no development qualifiers; the holdout is not applicable")
    if frame is None or inputs is None:
        raise Phase10AError("development qualifiers exist; the research frame and inputs are required")
    _check_inputs(reg, inputs)
    results_dir = Path(results_dir) if results_dir is not None else registry_path.parent
    by_name = {c.name: c for c in spec.candidates()}
    started, confirmed = datetime.now(timezone.utc).isoformat(timespec="seconds"), []
    for exp_id in qualifiers:
        entry = reg["results"][exp_id]
        cand = by_name[entry["model"]]
        cands = [c for c in spec.candidates() if c.task == cand.task and c.is_baseline] + [cand]
        cols = list(spec.feature_sets[entry["feature_set"]]["columns"])
        h = entry["horizon"]
        log(f"[holdout] {exp_id} ...")
        out = evaluate_excess(frame, cols, cands, horizon=h, n_splits=spec.n_splits,
                              splits=lambda ev: holdout_splits(ev["Date"], h, spec.n_splits,
                                                               spec.holdout_start, spec.holdout_end))
        artifact = _write_new_json(results_dir / "experiments" / f"holdout_{exp_id}.json",
                                   {"experiment_id": exp_id, "inputs": inputs, **out})
        g = out["models"][-1]["qualification"]
        ok = entry["development"]["status"] == QUALIFIED and g["status"] == QUALIFIED
        if ok:
            confirmed.append(exp_id)
        entry["holdout"] = {"oos_period": out["evaluation_period"], "n_oos_rows": out["evaluation_period"]["n_oos_rows"],
                            "metrics": out["models"][-1]["metrics"], "baseline_metrics": out["baseline_metrics"],
                            "gate": g, "status": g["status"], "p_value_raw": g["diebold_mariano"]["p_value"],
                            "artifact": artifact}
        entry["status"] = "CONFIRMED" if ok else "NOT_CONFIRMED"
    reg["holdout"] = {"status": "COMPLETE", "started_at": started,
                      "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "evaluated": qualifiers, "confirmed": confirmed,
                      "result": f"CONFIRMED: {confirmed}" if confirmed else NO_CONFIRMED_MODEL}
    reg["status"] = "COMPLETE"
    _write_json_atomic(registry_path, reg)
    return reg


# ==========================================
# CLI
# ==========================================


def _status_lines(reg: dict) -> list[str]:
    m = reg["matrix"]
    lines = [f"Registry : {reg['status']}  matrix sha256 {reg['matrix_sha256']}",
             f"Matrix   : {len(m['experiments'])} experiments = {len(m['feature_sets'])} feature sets x "
             f"{len(m['horizons'])} horizons x {sum(not c['is_baseline'] for c in m['candidates'])} models "
             f"({sum(e['task'] == REGRESSION for e in m['experiments'])} regression / "
             f"{sum(e['task'] == CLASSIFICATION for e in m['experiments'])} classification)",
             "Features : " + ", ".join(f"{k} {v['n_features']}" for k, v in m["feature_sets"].items())]
    if reg.get("development"):
        d = reg["development"]
        lines.append(f"Develop  : {d['result']}  ({d['n_tests']} tests; raw p<0.05 "
                     f"{d['multiple_comparisons']['n_raw_p_below_alpha']}; Holm p<0.05 "
                     f"{d['multiple_comparisons']['n_holm_p_below_alpha']})")
    if reg.get("holdout"):
        lines.append(f"Holdout  : {reg['holdout']['status']} - {reg['holdout']['result']}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 10A excess-return signal research (pre-registered)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "target-diagnostics", "develop", "holdout", "status"):
        p = sub.add_parser(name)
        p.add_argument("--registry", type=Path, default=REGISTRY_PATH)
        p.add_argument("--store", type=Path, default=DEFAULT_STORE)
        p.add_argument("--spy", type=Path, default=DEFAULT_SPY)
        p.add_argument("--qqq", type=Path, default=DEFAULT_QQQ)
        if name == "holdout":
            p.add_argument("--confirm", action="store_true", help="explicit authorization to run the holdout")
    args = parser.parse_args(argv)

    if args.command == "status":
        print("\n".join(_status_lines(load_registry(args.registry))))
        return 0
    if args.command == "holdout":
        if not args.confirm:
            raise Phase10AError("the Phase-10A confirmation holdout runs only with explicit authorization (--confirm)")
        reg = load_registry(args.registry)
        if reg["status"] != "DEVELOPMENT_COMPLETE" or not reg["development"]["qualifiers"]:
            raise Phase10AError(f"holdout not applicable (registry status {reg['status']})")

    inputs = collect_inputs(args.store, args.spy, args.qqq)
    frame, prov = build_frame(args.store, args.spy, args.qqq)
    if args.command == "freeze":
        rebuilt, _ = build_frame(args.store, args.spy, args.qqq)
        aapl, _ = load_raw_snapshot(PROJECT_ROOT / inputs["aapl_snapshot"]["path"])
        validation = freeze_validation(frame, prov, aapl, OFFICIAL, rebuilt_frame=rebuilt)
        reg = freeze(args.registry, inputs=inputs, validation=validation)
        print(f"Frozen   : {args.registry}")
        print("\n".join(_status_lines(reg)))
        return 0
    if args.command == "target-diagnostics":
        diag = run_target_diagnostics(args.registry, frame, inputs)
        for h, d in diag["horizons"].items():
            s = d["future_excess_return"]
            print(f"h={h}: {d['labelled_rows']} labelled dev rows; mean {s['mean']:.6f} std {s['std']:.6f}; "
                  f"positive {d['excess_direction']['positive_pct']:.2f}%; spot checks ok {d['all_spot_checks_match']}")
        return 0
    if args.command == "develop":
        print("\n".join(_status_lines(run_development(args.registry, frame, inputs))))
        return 0
    print("\n".join(_status_lines(run_holdout(args.registry, frame, inputs, confirm=True))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
