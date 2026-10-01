"""
Phase 10B - volatility-normalized excess-return research (AAPL vs SPY), pre-registered.

Question: does normalizing AAPL's future excess return over SPY by its
prediction-time realized volatility (training/normalized_targets.py) give a
target on which the existing features show significant out-of-sample
improvement over the baselines? A valid outcome is NO QUALIFIED MODEL.

Everything except the target is the frozen Phase 10A design: feature sets
A-F (F = 64 distinct, sw_count_1 dropped), the Phase 9 fixed candidates
(seed 42, no tuning), TimeSeriesSplit(20) with gap = horizon, gate_v1
unchanged, Holm informational, development 2017-02-01..2024-09-30,
confirmation holdout 2024-10-01..2026-09-25 (not pristine; evaluated only for
development qualifiers with --confirm, never read otherwise).

Harness use: targets come from build_normalized_targets; the harness's
validate_dataset, walk_forward, pooled_metrics and qualify are used unchanged
(as in Phase 10A). With volatility == 1 the pipeline reproduces Phase 10A's
evaluate_excess exactly (tested).

Note on the classification half: volatility > 0, so the normalized excess
direction equals the Phase 10A excess direction; with the same rows, folds,
features and fixed models the 90 classification experiments are expected to
reproduce Phase 10A's classification results. They are kept because they
are part of the pre-registered 198-experiment matrix.

Usage (project root):
    python -m training.phase10b_research freeze
    python -m training.phase10b_research target-diagnostics
    python -m training.phase10b_research develop
    python -m training.phase10b_research holdout --confirm
    python -m training.phase10b_research status
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

import training.phase10a_research as p10a  # noqa: E402
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
from training.normalized_targets import (  # noqa: E402
    HORIZONS,
    NORMALIZED_TARGET_COLUMNS,
    SPY_CLOSE,
    VOL_COLUMN,
    VOLATILITY_DEFINITION,
    attach_volatility,
    build_normalized_targets,
    target_definition,
)
from training.phase9_research import (  # noqa: E402
    DEFAULT_STORE,
    ResearchError,
    _slug,
    _write_json_atomic,
    _write_new_json,
    candidate_spec,
    canonical_json,
    environment,
    holdout_splits,
    holm_adjust,
    phase9_candidates,
    walk_forward_with_splits,
)
from training.targets import target_summary  # noqa: E402

PHASE = "10B"
MATRIX_VERSION = "phase10b_matrix_v1"
REGISTRY_VERSION = "phase10b_registry_v1"
SYMBOL = "AAPL"
N_SPLITS = p10a.N_SPLITS
DEV_START, DEV_END = p10a.DEV_START, p10a.DEV_END
HOLDOUT_START, HOLDOUT_END = p10a.HOLDOUT_START, p10a.HOLDOUT_END

RESEARCH_DIR = PROJECT_ROOT / "data" / "results" / "research" / "phase10b"
REGISTRY_PATH = RESEARCH_DIR / "registry.json"
DEFAULT_SPY, DEFAULT_QQQ = p10a.DEFAULT_SPY, p10a.DEFAULT_QQQ

NO_DEVELOPMENT_QUALIFIERS = p10a.NO_DEVELOPMENT_QUALIFIERS
NO_CONFIRMED_MODEL = p10a.NO_CONFIRMED_MODEL
DETERMINISM_NOT_VERIFIED = p10a.DETERMINISM_NOT_VERIFIED

FEATURE_SETS = p10a.FEATURE_SETS                       # frozen Phase 10A feature matrix, reused unchanged
DUPLICATE_RESOLUTION = p10a.DUPLICATE_RESOLUTION
ALL_FEATURE_COLUMNS = p10a.ALL_FEATURE_COLUMNS

BASELINE_LABELS = {
    "Mean Return": "Mean Normalized Excess Return - training-fold mean of normalized_excess_return_h",
    "Zero Return": "Zero Normalized Excess Return - constant 0",
    "Base Rate": "Normalized Excess Direction Base Rate - training-fold share of normalized_excess_direction_h == 1",
    "Always UP": "Always UP - always predicts normalized_excess_direction_h = 1 with probability 1",
}


class Phase10BError(ResearchError):
    """The Phase 10B protocol would be violated."""


def experiment_id(feature_set: str, horizon: int, model: str) -> str:
    return f"p10b_{feature_set}_h{horizon}_{_slug(model)}"


@dataclass(frozen=True)
class Spec10B:
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
            raise Phase10BError(f"candidate names must be unique: {names}")
        return cands

    def matrix(self) -> dict:
        cands = self.candidates()
        for fs, d in self.feature_sets.items():
            cols = list(d["columns"])
            if len(set(cols)) != len(cols):
                raise Phase10BError(f"feature set {fs} lists a column twice")
            leaked = sorted(set(cols) & set(NORMALIZED_TARGET_COLUMNS))
            if leaked:
                raise Phase10BError(f"feature set {fs} contains target/price/volatility columns {leaked}")
        experiments = [{"experiment_id": experiment_id(fs, h, c.name), "feature_set": fs, "horizon": h,
                        "model": c.name, "task": c.task}
                       for fs in self.feature_sets for h in self.horizons for c in cands if not c.is_baseline]
        base = p10a.Spec10A(feature_sets=self.feature_sets, horizons=self.horizons,
                            candidates_factory=self.candidates_factory, n_splits=self.n_splits,
                            dev_start=self.dev_start, dev_end=self.dev_end, holdout_start=self.holdout_start,
                            holdout_end=self.holdout_end).matrix()
        evaluation = dict(base["evaluation"])
        evaluation["target_construction"] = ("training.normalized_targets.build_normalized_targets replaces "
                                             "evaluation_harness.build_targets (excess return / prediction-time "
                                             "volatility)")
        evaluation["baselines"] = {**evaluation["baselines"], "report_labels": BASELINE_LABELS}
        evaluation["confirmation_holdout"] = {**evaluation["confirmation_holdout"],
                                              "name": "Phase-10B confirmation holdout"}
        return {
            "phase": PHASE,
            "matrix_version": MATRIX_VERSION,
            "symbol": SYMBOL,
            "benchmark": "SPY",
            "research_question": ("Does normalizing AAPL's future excess return relative to SPY by prediction-time "
                                  "realized volatility improve out-of-sample predictive performance?"),
            "hypothesis": "a volatility-normalized excess target may be more learnable (to be tested, not assumed)",
            "reused_design": "Phase 10A feature sets, candidates, evaluation and dates (matrix "
                             "fbd763e7980232bb1bf7409616f8dbbb0af9e8b0c70857a680cb89aad1da6a45)",
            "feature_sets": {k: {"families": list(v["families"]), "columns": list(v["columns"]),
                                 "n_features": len(v["columns"])} for k, v in self.feature_sets.items()},
            "f_duplicate_resolution": DUPLICATE_RESOLUTION,
            "feature_definitions": base["feature_definitions"],
            "horizons": list(self.horizons),
            "targets": {str(h): target_definition(h) for h in self.horizons},
            "volatility_definition": VOLATILITY_DEFINITION,
            "classification_note": ("volatility > 0, so normalized_excess_direction_h equals the Phase 10A "
                                    "excess_direction_h; the 90 classification experiments are expected to "
                                    "reproduce Phase 10A"),
            "candidates": [{**candidate_spec(c), "report_label": BASELINE_LABELS.get(c.name, c.name)} for c in cands],
            "evaluation": evaluation,
            "leakage_rules": [*base["leakage_rules"],
                              "volatility denominator: closes dated <= D only; never a feature"],
            "experiments": experiments,
        }


OFFICIAL = Spec10B()


def matrix_sha256(matrix: dict) -> str:
    return hashlib.sha256(canonical_json(matrix).encode("utf-8")).hexdigest()


# ==========================================
# Frame, validation
# ==========================================


def collect_inputs(store_path: Path = DEFAULT_STORE, spy_path: Path = DEFAULT_SPY, qqq_path: Path = DEFAULT_QQQ) -> dict:
    return p10a.collect_inputs(store_path, spy_path, qqq_path)


def build_frame(store_path: Path = DEFAULT_STORE, spy_path: Path = DEFAULT_SPY,
                qqq_path: Path = DEFAULT_QQQ) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Phase 10A frame (+ SPY_Close) + prediction-time excess volatility from the verified snapshots."""
    frame, prov = p10a.build_frame(store_path, spy_path, qqq_path)
    inputs = p10a.input_provenance(store_path)
    aapl, _ = load_raw_snapshot(PROJECT_ROOT / inputs["aapl_snapshot"]["path"])
    spy, _ = mc.load_context_snapshot(spy_path, "SPY")
    return attach_volatility(frame, aapl, spy), prov, aapl


def frame_sha256(frame: pd.DataFrame) -> str:
    cols = ["Date", "Close", SPY_CLOSE, VOL_COLUMN, *ALL_FEATURE_COLUMNS]
    return hashlib.sha256(frame[cols].to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def development_frame(frame: pd.DataFrame, spec: Spec10B = OFFICIAL) -> pd.DataFrame:
    return p10a.development_frame(frame, spec)


def freeze_validation(frame: pd.DataFrame, prov: dict, aapl_bars: pd.DataFrame, spec: Spec10B = OFFICIAL,
                      rebuilt_frame: pd.DataFrame | None = None) -> dict:
    """Phase 10A structural checks + volatility and normalized-target checks (development targets only)."""
    out = p10a.freeze_validation(frame, prov, aapl_bars, spec, rebuilt_frame=None)
    vol = frame[VOL_COLUMN].to_numpy(dtype="float64")
    out["volatility"] = {"rows": len(vol), "finite_positive": bool(np.isfinite(vol).all() and (vol > 0).all())}
    dev = development_frame(frame, spec)
    det = out["construction_determinism"]
    if rebuilt_frame is not None:
        det["frame_sha256"] = frame_sha256(frame)
        det["frame_rebuild_identical"] = frame_sha256(frame) == frame_sha256(rebuilt_frame)
    norm = {}
    for h in spec.horizons:
        cols = ["Date", "Close", SPY_CLOSE, VOL_COLUMN]
        t1, t2 = build_normalized_targets(dev[cols], h), build_normalized_targets(dev[cols], h)
        norm[str(h)] = (hashlib.sha256(t1.to_csv(index=False).encode()).hexdigest()
                        == hashlib.sha256(t2.to_csv(index=False).encode()).hexdigest())
    det["development_normalized_targets_identical"] = norm
    out["leakage"]["normalized_target_columns_in_any_feature_set"] = sorted(
        set(ALL_FEATURE_COLUMNS) & set(NORMALIZED_TARGET_COLUMNS))
    return out


# ==========================================
# Registry
# ==========================================


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    path = Path(path)
    if not path.is_file():
        raise Phase10BError(f"registry not found: {path} (run `freeze` first)")
    reg = json.loads(path.read_text(encoding="utf-8"))
    if reg.get("registry_version") != REGISTRY_VERSION or reg.get("phase") != PHASE:
        raise Phase10BError("not a Phase 10B registry")
    if matrix_sha256(reg["matrix"]) != reg["matrix_sha256"]:
        raise Phase10BError("registry matrix does not match its recorded hash (edited after freezing)")
    return reg


def _check_spec(reg: dict, spec: Spec10B) -> dict:
    matrix = spec.matrix()
    if matrix_sha256(matrix) != reg["matrix_sha256"]:
        raise Phase10BError("the experiment matrix in code differs from the FROZEN matrix")
    return matrix


def _check_inputs(reg: dict, inputs: dict) -> None:
    if inputs != reg["inputs"]:
        changed = sorted(k for k in set(inputs) | set(reg["inputs"]) if inputs.get(k) != reg["inputs"].get(k))
        raise Phase10BError(f"input hashes differ from the frozen registry: {changed} - stopping (nothing regenerated)")


def freeze(registry_path: Path = REGISTRY_PATH, *, inputs: dict, validation: dict, spec: Spec10B = OFFICIAL) -> dict:
    registry_path = Path(registry_path)
    if registry_path.exists():
        raise Phase10BError(f"{registry_path} already exists; the Phase 10B matrix is frozen once")
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


def target_diagnostics(frame: pd.DataFrame, spec: Spec10B = OFFICIAL) -> dict:
    dev = development_frame(frame, spec)
    out = {"scope": "development rows only (<= development end); holdout: structural counts only",
           "development_rows": len(dev), "horizons": {}, "volatility": p10a._stats(dev[VOL_COLUMN])}
    close, spy, vol = dev["Close"].to_numpy(), dev[SPY_CLOSE].to_numpy(), dev[VOL_COLUMN].to_numpy()
    for h in spec.horizons:
        t = build_normalized_targets(dev[["Date", "Close", SPY_CLOSE, VOL_COLUMN]], h)
        n = len(t)
        picks = sorted({0, 1, n // 4, n // 2, (3 * n) // 4, n - 2, n - 1})
        spot = []
        for i in picks:                                   # independent recomputation from raw values
            e = (close[i + h] / close[i] - 1) - (spy[i + h] / spy[i] - 1)
            z = e / vol[i]
            spot.append({"row": int(i), "date": t["Date"].iloc[i].date().isoformat(),
                         "label_close_date": dev["Date"].iloc[i + h].date().isoformat(),
                         "excess": float(e), "volatility": float(vol[i]), "normalized": float(z),
                         "matches": bool(abs(z - t["future_return"].iloc[i]) <= 1e-12 * max(1.0, abs(z))
                                         and t["direction"].iloc[i] == int(z > 0))})
        out["horizons"][str(h)] = {
            "target": target_definition(h)["regression"]["name"],
            "labelled_rows": n, "unlabelled_tail_rows": h,
            "last_label_close_date": dev["Date"].iloc[n - 1 + h].date().isoformat(),
            "normalized_excess_return": p10a._stats(t["future_return"]),
            "future_excess_return": p10a._stats(t["future_excess_return"]),
            "normalized_excess_direction": {**target_summary(t["future_return"], t["direction"]),
                                            "positive_pct": float(100 * t["direction"].mean()),
                                            "non_positive_pct": float(100 * (1 - t["direction"].mean()))},
            "direction_equals_excess_direction": bool((t["direction"] == (t["future_excess_return"] > 0)
                                                       .astype(int)).all()),
            "spot_checks": spot, "all_spot_checks_match": all(s["matches"] for s in spot),
            "holdout_structural": p10a.holdout_label_counts(frame, h, spec),
        }
    out["target_columns_in_any_feature_set"] = sorted(set(ALL_FEATURE_COLUMNS) & set(NORMALIZED_TARGET_COLUMNS))
    return out


def run_target_diagnostics(registry_path: Path, frame: pd.DataFrame, inputs: dict, *, spec: Spec10B = OFFICIAL,
                           results_dir: Path | None = None) -> dict:
    registry_path = Path(registry_path)
    reg = load_registry(registry_path)
    _check_spec(reg, spec)
    _check_inputs(reg, inputs)
    if reg["target_diagnostics"] is not None or reg["development"] is not None:
        raise Phase10BError("target diagnostics already recorded (or development already run)")
    results_dir = Path(results_dir) if results_dir is not None else registry_path.parent
    diag = target_diagnostics(frame, spec)
    artifact = _write_new_json(results_dir / "targets" / "target_diagnostics_development.json", diag)
    reg["target_diagnostics"] = {"artifact": artifact,
                                 "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _write_json_atomic(registry_path, reg)
    return diag


# ==========================================
# Evaluation (harness components, normalized targets)
# ==========================================


def evaluate_normalized(frame: pd.DataFrame, feature_columns: list[str], candidates: list[Candidate], *,
                        horizon: int, n_splits: int, splits=None) -> dict:
    data = frame[["Date", "Close", SPY_CLOSE, VOL_COLUMN, *feature_columns]].copy()
    validate_dataset(data, feature_columns)
    evaluable = build_normalized_targets(data, horizon)
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


def run_development(registry_path: Path, frame: pd.DataFrame, inputs: dict, *, spec: Spec10B = OFFICIAL,
                    results_dir: Path | None = None, evaluate=evaluate_normalized, log=print) -> dict:
    registry_path = Path(registry_path)
    reg = load_registry(registry_path)
    if reg["status"] != "FROZEN" or reg["development"] is not None:
        raise Phase10BError("development has already been run for this registry")
    matrix = _check_spec(reg, spec)
    _check_inputs(reg, inputs)
    results_dir = Path(results_dir) if results_dir is not None else registry_path.parent
    exp_dir = results_dir / "experiments"
    planned = [exp_dir / f"dev_{fs}_h{h}.json" for fs in spec.feature_sets for h in spec.horizons]
    if any(p.exists() for p in planned):
        raise Phase10BError(f"development artifacts already exist in {exp_dir}")

    dev = development_frame(frame, spec)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    specs = {c["name"]: c for c in matrix["candidates"]}
    entries, runs = [], []
    for fs, fdef in spec.feature_sets.items():
        cols = list(fdef["columns"])
        for h in spec.horizons:
            name = f"phase10b_dev_{fs}_h{h}"
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
        raise Phase10BError("development results do not cover exactly the pre-registered experiments")
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
        reg["status"] = "DEVELOPMENT_COMPLETE"
    else:
        reg["holdout"] = {"status": "NOT_APPLICABLE", "result": NO_DEVELOPMENT_QUALIFIERS, "recorded_at": now,
                          "explanation": "no development qualifier; the Phase-10B confirmation holdout was not read",
                          "confirmed": []}
        reg["status"] = "COMPLETE"
    _write_json_atomic(registry_path, reg)
    return reg


# ==========================================
# Confirmation holdout (qualifiers + explicit authorization only)
# ==========================================


def run_holdout(registry_path: Path, frame: pd.DataFrame | None, inputs: dict | None, *, confirm: bool,
                spec: Spec10B = OFFICIAL, results_dir: Path | None = None, log=print) -> dict:
    if not confirm:
        raise Phase10BError("the Phase-10B confirmation holdout runs only with explicit authorization (--confirm)")
    registry_path = Path(registry_path)
    reg = load_registry(registry_path)
    if reg["status"] != "DEVELOPMENT_COMPLETE" or reg["holdout"] is not None:
        raise Phase10BError("holdout requires development qualifiers and runs exactly once "
                            f"(registry status {reg['status']})")
    _check_spec(reg, spec)
    qualifiers = list(reg["development"]["qualifiers"])
    if not qualifiers:
        raise Phase10BError("no development qualifiers; the holdout is not applicable")
    if frame is None or inputs is None:
        raise Phase10BError("development qualifiers exist; the research frame and inputs are required")
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
        out = evaluate_normalized(frame, cols, cands, horizon=h, n_splits=spec.n_splits,
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
    parser = argparse.ArgumentParser(description="Phase 10B normalized excess-return research (pre-registered)")
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
            raise Phase10BError("the Phase-10B confirmation holdout runs only with explicit authorization (--confirm)")
        reg = load_registry(args.registry)
        if reg["status"] != "DEVELOPMENT_COMPLETE" or not reg["development"]["qualifiers"]:
            raise Phase10BError(f"holdout not applicable (registry status {reg['status']})")

    inputs = collect_inputs(args.store, args.spy, args.qqq)
    frame, prov, aapl = build_frame(args.store, args.spy, args.qqq)
    if args.command == "freeze":
        rebuilt, _, _ = build_frame(args.store, args.spy, args.qqq)
        validation = freeze_validation(frame, prov, aapl, OFFICIAL, rebuilt_frame=rebuilt)
        reg = freeze(args.registry, inputs=inputs, validation=validation)
        print(f"Frozen   : {args.registry}")
        print("\n".join(_status_lines(reg)))
        return 0
    if args.command == "target-diagnostics":
        diag = run_target_diagnostics(args.registry, frame, inputs)
        v = diag["volatility"]
        print(f"volatility (dev): mean {v['mean']:.6f} median {v['median']:.6f} min {v['min']:.6f} max {v['max']:.6f}")
        for h, d in diag["horizons"].items():
            s = d["normalized_excess_return"]
            print(f"h={h}: {d['labelled_rows']} labelled dev rows; normalized mean {s['mean']:.4f} std {s['std']:.4f} "
                  f"min {s['min']:.3f} max {s['max']:.3f}; positive {d['normalized_excess_direction']['positive_pct']:.2f}%; "
                  f"spot checks ok {d['all_spot_checks_match']}; direction == excess direction "
                  f"{d['direction_equals_excess_direction']}")
        return 0
    if args.command == "develop":
        print("\n".join(_status_lines(run_development(args.registry, frame, inputs))))
        return 0
    print("\n".join(_status_lines(run_holdout(args.registry, frame, inputs, confirm=True))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
