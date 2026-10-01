"""
Phase 10C - market-regime-conditioned research (AAPL vs SPY excess return), pre-registered.

Question: do predictive relationships emerge conditionally under observable
market regimes although no unconditional AAPL model qualified in Phases
9 / 10A / 10B? A valid outcome is NO DEVELOPMENT QUALIFIERS.

Design (frozen before any result):
  target       Phase 10A excess return (training/excess_targets.py)
  matrix       the frozen Phase 10A feature sets A-F x horizons 1/3/5 x the 11
               Phase 9 fixed models = 198 base experiments (18 walk-forward runs)
  regimes      training/regimes.py: R1 AAPL vol, R2 SPY vol, R3 SPY trend,
               R4 SPY 20-session return, R5 SPY vol x SPY trend = 12 states
  evaluation   ONE model per training fold (harness walk_forward,
               TimeSeriesSplit(20), gap = h); out-of-sample rows labelled with
               thresholds fitted on that fold's training rows; for each state
               the unchanged pooled_metrics + qualify (gate_v1) run on exactly
               the state's OOS rows for all candidates AND baselines.
               "gate_v1 logic unchanged; evaluation population is the
               pre-registered regime-conditioned validation population."
  sample rule  a state with < 100 pooled OOS rows -> INSUFFICIENT_DATA, no test
  family       all eligible (state >= 100 rows) regime-conditioned comparisons,
               at most 198 x 12 = 2,376; Holm across the complete family
  qualifies    gate_v1 QUALIFIED within the state AND rows >= 100 AND raw p < 0.05
               AND Holm-adjusted p < 0.05
  holdout      2024-10-01 .. 2026-09-25 (not pristine) only for qualifiers, only
               with --confirm; otherwise NOT_APPLICABLE and never read

Usage (project root):
    python -m training.phase10c_research freeze
    python -m training.phase10c_research regime-diagnostics
    python -m training.phase10c_research develop
    python -m training.phase10c_research holdout --confirm
    python -m training.phase10c_research status
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
from training import regimes as rg  # noqa: E402
from training.build_dataset import file_sha256, load_raw_snapshot  # noqa: E402
from training.evaluation_harness import (  # noqa: E402
    CLASSIFICATION,
    GATE_VERSION,
    HARNESS_VERSION,
    QUALIFIED,
    REGRESSION,
    Candidate,
    HarnessConfig,
    git_info,
    make_splitter,
    pooled_metrics,
    qualify,
    validate_dataset,
    walk_forward,
)
from training.excess_targets import HORIZONS, SPY_CLOSE, TARGET_COLUMNS, build_excess_targets, target_definition  # noqa: E402
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

PHASE = "10C"
MATRIX_VERSION = "phase10c_matrix_v1"
REGISTRY_VERSION = "phase10c_registry_v1"
SYMBOL = "AAPL"
N_SPLITS = p10a.N_SPLITS
DEV_START, DEV_END = p10a.DEV_START, p10a.DEV_END
HOLDOUT_START, HOLDOUT_END = p10a.HOLDOUT_START, p10a.HOLDOUT_END
MIN_STATE_OOS = 100
ALPHA = HarnessConfig().significance_level
HOLM_ALPHA = 0.05

RESEARCH_DIR = PROJECT_ROOT / "data" / "results" / "research" / "phase10c"
REGISTRY_PATH = RESEARCH_DIR / "registry.json"
DEFAULT_SPY, DEFAULT_QQQ = p10a.DEFAULT_SPY, p10a.DEFAULT_QQQ

NO_DEVELOPMENT_QUALIFIERS = p10a.NO_DEVELOPMENT_QUALIFIERS
NO_CONFIRMED_MODEL = p10a.NO_CONFIRMED_MODEL
DETERMINISM_NOT_VERIFIED = p10a.DETERMINISM_NOT_VERIFIED
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

FEATURE_SETS = p10a.FEATURE_SETS
ALL_FEATURE_COLUMNS = p10a.ALL_FEATURE_COLUMNS
BASELINE_LABELS = p10a.BASELINE_LABELS
DM_LIMITATION = ("Within regime-conditioned subsets, observations are not necessarily consecutive in "
                 "calendar/session time. The existing Newey-West correction therefore treats the pooled "
                 "regime-filtered sequence as consecutive observations. This approximation is retained to "
                 "preserve gate_v1 comparability with Phases 9-10B.")


class Phase10CError(ResearchError):
    """The Phase 10C protocol would be violated."""


def base_experiment_id(fs: str, h: int, model: str) -> str:
    return f"p10c_{fs}_h{h}_{_slug(model)}"


def comparison_id(fs: str, h: int, model: str, family: str, state: str) -> str:
    return f"{base_experiment_id(fs, h, model)}__{family}__{state}"


@dataclass(frozen=True)
class Spec10C:
    feature_sets: dict = field(default_factory=lambda: FEATURE_SETS)
    horizons: tuple = HORIZONS
    candidates_factory: Callable[[], list[Candidate]] = phase9_candidates
    n_splits: int = N_SPLITS
    dev_start: date = DEV_START
    dev_end: date = DEV_END
    holdout_start: date = HOLDOUT_START
    holdout_end: date = HOLDOUT_END
    min_state_oos: int = MIN_STATE_OOS

    def candidates(self) -> list[Candidate]:
        cands = self.candidates_factory()
        names = [c.name for c in cands]
        if len(set(names)) != len(names):
            raise Phase10CError(f"candidate names must be unique: {names}")
        return cands

    def matrix(self) -> dict:
        cands = self.candidates()
        for fs, d in self.feature_sets.items():
            cols = list(d["columns"])
            bad = sorted(set(cols) & (set(TARGET_COLUMNS) | set(rg.REGIME_VARIABLE_COLUMNS)))
            if len(set(cols)) != len(cols) or bad:
                raise Phase10CError(f"feature set {fs} invalid (duplicates or target/regime columns {bad})")
        base = [{"experiment_id": base_experiment_id(fs, h, c.name), "feature_set": fs, "horizon": h,
                 "model": c.name, "task": c.task}
                for fs in self.feature_sets for h in self.horizons for c in cands if not c.is_baseline]
        comparisons = [comparison_id(e["feature_set"], e["horizon"], e["model"], fam, st)
                       for e in base for fam, fd in rg.FAMILIES.items() for st in fd["states"]]
        a = p10a.Spec10A(feature_sets=self.feature_sets, horizons=self.horizons,
                         candidates_factory=self.candidates_factory, n_splits=self.n_splits,
                         dev_start=self.dev_start, dev_end=self.dev_end, holdout_start=self.holdout_start,
                         holdout_end=self.holdout_end).matrix()
        evaluation = dict(a["evaluation"])
        evaluation.update({
            "population": ("gate_v1 logic unchanged; evaluation population is the pre-registered "
                           "regime-conditioned validation population"),
            "model_strategy": "one model per training fold (walk_forward); predictions evaluated per regime state",
            "baselines_population": "baselines scored on exactly the same regime-filtered OOS rows",
            "dm_limitation": DM_LIMITATION,
            "min_state_oos_rows": self.min_state_oos,
            "insufficient_rule": f"state with < {self.min_state_oos} pooled OOS rows -> {INSUFFICIENT_DATA}, no test",
            "multiple_comparisons": {"method": "Holm", "family": ("all eligible regime-conditioned comparisons of "
                                                                  "Phase 10C (states with >= min rows)"),
                                     "max_family_size": len(comparisons), "alpha": HOLM_ALPHA,
                                     "role": "part of the Phase 10C qualification criterion"},
            "qualification_criterion": ("gate_v1 QUALIFIED within the state AND state OOS rows >= "
                                        f"{self.min_state_oos} AND raw DM p < {ALPHA} AND Holm-adjusted p < {HOLM_ALPHA}"),
            "unconditional_reference": "pooled all-OOS result recorded for reference only (not in the family)",
            "confirmation_holdout": {**evaluation["confirmation_holdout"], "name": "Phase-10C confirmation holdout",
                                     "confirmation_rule": ("qualifier confirmed iff gate_v1 QUALIFIED in the same "
                                                           f"state on the holdout with >= {self.min_state_oos} rows; "
                                                           "thresholds fitted on each holdout fold's training rows")},
        })
        return {
            "phase": PHASE, "matrix_version": MATRIX_VERSION, "symbol": SYMBOL, "benchmark": "SPY",
            "research_question": ("Do predictive relationships emerge conditionally under observable market regimes "
                                  "even though no unconditional AAPL model qualified in Phases 9, 10A or 10B?"),
            "hypothesis": ("explicitly pre-registered regime conditioning may reveal statistically reliable "
                           "out-of-sample performance absent in the unconditional evaluation (to be tested)"),
            "reused_design": "Phase 10A target, feature sets, candidates, evaluation and dates (matrix "
                             "fbd763e7980232bb1bf7409616f8dbbb0af9e8b0c70857a680cb89aad1da6a45)",
            "feature_sets": {k: {"families": list(v["families"]), "columns": list(v["columns"]),
                                 "n_features": len(v["columns"])} for k, v in self.feature_sets.items()},
            "f_duplicate_resolution": p10a.DUPLICATE_RESOLUTION,
            "horizons": list(self.horizons),
            "targets": {str(h): target_definition(h) for h in self.horizons},
            "regimes": {"families": rg.FAMILIES, "variables": rg.VARIABLE_DEFINITIONS, "state_count": rg.STATE_COUNT,
                        "threshold_fitting": ("fold_median thresholds = median of the variable over the TRAINING rows "
                                              "of each walk-forward fold; validation rows labelled with that frozen "
                                              "threshold; ties (== median) -> LOW")},
            "candidates": [{**candidate_spec(c), "report_label": BASELINE_LABELS.get(c.name, c.name)} for c in cands],
            "evaluation": evaluation,
            "leakage_rules": [*a["leakage_rules"],
                              "regime variables use bars dated <= D only; regime thresholds from training rows only"],
            "base_experiments": base,
            "n_base_experiments": len(base),
            "comparisons": comparisons,
            "n_comparisons_max": len(comparisons),
        }


OFFICIAL = Spec10C()


def matrix_sha256(matrix: dict) -> str:
    return hashlib.sha256(canonical_json(matrix).encode("utf-8")).hexdigest()


# ==========================================
# Frame
# ==========================================


def collect_inputs(store_path: Path = DEFAULT_STORE, spy_path: Path = DEFAULT_SPY, qqq_path: Path = DEFAULT_QQQ) -> dict:
    return p10a.collect_inputs(store_path, spy_path, qqq_path)


def build_frame(store_path: Path = DEFAULT_STORE, spy_path: Path = DEFAULT_SPY,
                qqq_path: Path = DEFAULT_QQQ) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    frame, prov = p10a.build_frame(store_path, spy_path, qqq_path)
    inputs = p10a.input_provenance(store_path)
    aapl, _ = load_raw_snapshot(PROJECT_ROOT / inputs["aapl_snapshot"]["path"])
    spy, _ = mc.load_context_snapshot(spy_path, "SPY")
    return rg.attach_regime_variables(frame, aapl, spy), prov, aapl


def frame_sha256(frame: pd.DataFrame) -> str:
    cols = ["Date", "Close", SPY_CLOSE, *rg.REGIME_VARIABLE_COLUMNS, *ALL_FEATURE_COLUMNS]
    return hashlib.sha256(frame[cols].to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def development_frame(frame: pd.DataFrame, spec: Spec10C = OFFICIAL) -> pd.DataFrame:
    return p10a.development_frame(frame, spec)


def _values(evaluable: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c: evaluable[c].to_numpy(dtype="float64") for c in rg.REGIME_INPUT_COLUMNS}


def freeze_validation(frame: pd.DataFrame, prov: dict, aapl_bars: pd.DataFrame, spec: Spec10C = OFFICIAL,
                      rebuilt_frame: pd.DataFrame | None = None) -> dict:
    out = p10a.freeze_validation(frame, prov, aapl_bars, spec, rebuilt_frame=None)
    diff = float(np.max(np.abs(frame[rg.AAPL_VOL].to_numpy() - frame["Volatility"].to_numpy())))
    out["regimes"] = {
        "variables_finite": bool(np.isfinite(frame[list(rg.REGIME_INPUT_COLUMNS)].to_numpy(dtype="float64")).all()),
        "aapl_vol_equals_technical_Volatility_max_abs_diff": diff,
        "regime_variable_columns_in_feature_sets": sorted(set(ALL_FEATURE_COLUMNS) & set(rg.REGIME_VARIABLE_COLUMNS)),
    }
    if diff > 1e-12:
        raise Phase10CError(f"regime_aapl_vol_20 differs from the technical Volatility feature ({diff})")
    if rebuilt_frame is not None:
        det = out["construction_determinism"]
        det["frame_sha256"] = frame_sha256(frame)
        det["frame_rebuild_identical"] = frame_sha256(frame) == frame_sha256(rebuilt_frame)
    return out


# ==========================================
# Registry
# ==========================================


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    path = Path(path)
    if not path.is_file():
        raise Phase10CError(f"registry not found: {path} (run `freeze` first)")
    reg = json.loads(path.read_text(encoding="utf-8"))
    if reg.get("registry_version") != REGISTRY_VERSION or reg.get("phase") != PHASE:
        raise Phase10CError("not a Phase 10C registry")
    if matrix_sha256(reg["matrix"]) != reg["matrix_sha256"]:
        raise Phase10CError("registry matrix does not match its recorded hash (edited after freezing)")
    return reg


def _check_spec(reg: dict, spec: Spec10C) -> dict:
    matrix = spec.matrix()
    if matrix_sha256(matrix) != reg["matrix_sha256"]:
        raise Phase10CError("the experiment matrix in code differs from the FROZEN matrix")
    return matrix


def _check_inputs(reg: dict, inputs: dict) -> None:
    if inputs != reg["inputs"]:
        changed = sorted(k for k in set(inputs) | set(reg["inputs"]) if inputs.get(k) != reg["inputs"].get(k))
        raise Phase10CError(f"input hashes differ from the frozen registry: {changed} - stopping (nothing regenerated)")


def freeze(registry_path: Path = REGISTRY_PATH, *, inputs: dict, validation: dict, spec: Spec10C = OFFICIAL) -> dict:
    registry_path = Path(registry_path)
    if registry_path.exists():
        raise Phase10CError(f"{registry_path} already exists; the Phase 10C matrix is frozen once")
    matrix = spec.matrix()
    reg = {"registry_version": REGISTRY_VERSION, "phase": PHASE, "matrix": matrix, "matrix_sha256": matrix_sha256(matrix),
           "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "frozen_git": git_info(),
           "frozen_environment": environment(), "inputs": inputs, "freeze_validation": validation,
           "status": "FROZEN", "regime_diagnostics": None, "development": None, "holdout": None,
           "determinism": DETERMINISM_NOT_VERIFIED, "results": None}
    _write_json_atomic(registry_path, reg)
    return reg


# ==========================================
# Regime diagnostics (development, feature-only: no target, no model)
# ==========================================


def regime_diagnostics(frame: pd.DataFrame, spec: Spec10C = OFFICIAL) -> dict:
    dev = development_frame(frame, spec)
    out = {"scope": "development rows only; uses regime variables and fold structure only (no target values, no models)",
           "development_rows": len(dev),
           "variables": {c: p10a._stats(dev[c]) for c in rg.REGIME_INPUT_COLUMNS}, "horizons": {}}
    for h in spec.horizons:
        n = len(dev) - h                                   # labelled rows; the last h rows have no label
        rows = dev.iloc[:n]
        splits = list(make_splitter(HarnessConfig(horizon=h, n_splits=spec.n_splits)).split(np.zeros((n, 1))))
        states, folds = rg.assign_oos_states(_values(rows), splits)
        counts = {fam: dict(Counter(states[fam].values())) for fam in rg.FAMILIES}
        eligible = {fam: {s: counts[fam].get(s, 0) >= spec.min_state_oos for s in fd["states"]}
                    for fam, fd in rg.FAMILIES.items()}
        out["horizons"][str(h)] = {
            "labelled_rows": n, "oos_rows": sum(f["n_test"] for f in folds),
            "pooled_oos_state_counts": {fam: {s: counts[fam].get(s, 0) for s in fd["states"]}
                                        for fam, fd in rg.FAMILIES.items()},
            "eligible_states": eligible,
            "n_eligible_states": sum(v for fam in eligible.values() for v in fam.values()),
            "folds": folds,
        }
    n_models = sum(not c.is_baseline for c in spec.candidates())
    out["expected_family_size"] = sum(len(spec.feature_sets) * n_models * out["horizons"][str(h)]["n_eligible_states"]
                                      for h in spec.horizons)
    return out


def run_regime_diagnostics(registry_path: Path, frame: pd.DataFrame, inputs: dict, *, spec: Spec10C = OFFICIAL,
                           results_dir: Path | None = None) -> dict:
    registry_path = Path(registry_path)
    reg = load_registry(registry_path)
    _check_spec(reg, spec)
    _check_inputs(reg, inputs)
    if reg["regime_diagnostics"] is not None or reg["development"] is not None:
        raise Phase10CError("regime diagnostics already recorded (or development already run)")
    results_dir = Path(results_dir) if results_dir is not None else registry_path.parent
    diag = regime_diagnostics(frame, spec)
    artifact = _write_new_json(results_dir / "regimes" / "regime_diagnostics_development.json", diag)
    reg["regime_diagnostics"] = {"artifact": artifact, "expected_family_size": diag["expected_family_size"],
                                 "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _write_json_atomic(registry_path, reg)
    return diag


# ==========================================
# Regime-conditioned evaluation
# ==========================================


def _subset_result(predictions: pd.DataFrame, rows: set[int], candidates: list[Candidate], config: HarnessConfig,
                   min_rows: int) -> dict:
    sub = predictions[predictions["row"].isin(rows)]
    n = int(sub["row"].nunique())
    if n < min_rows:
        return {"n_oos": n, "status": INSUFFICIENT_DATA}
    metrics = pooled_metrics(sub, candidates)
    gate = qualify(sub, candidates, metrics, config)
    return {"n_oos": n, "status": "TESTED",
            "models": [{"model": c.name, "task": c.task, "is_baseline": c.is_baseline, "metrics": metrics[c.name],
                        "qualification": gate[c.name]} for c in candidates]}


def evaluate_regimes(frame: pd.DataFrame, feature_columns: list[str], candidates: list[Candidate], *, horizon: int,
                     n_splits: int, min_rows: int = MIN_STATE_OOS, splits=None) -> dict:
    data = frame[["Date", "Close", SPY_CLOSE, *rg.REGIME_INPUT_COLUMNS, *feature_columns]].copy()
    data = data.loc[:, ~data.columns.duplicated()]
    validate_dataset(data, feature_columns)
    evaluable = build_excess_targets(data, horizon)
    config = HarnessConfig(horizon=horizon, n_splits=n_splits)
    split_list = (list(make_splitter(config).split(evaluable[feature_columns])) if splits is None
                  else list(splits(evaluable)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if splits is None:
            predictions, fold_metrics, folds = walk_forward(evaluable, feature_columns, candidates, config)
        else:
            predictions, fold_metrics, folds = walk_forward_with_splits(evaluable, feature_columns, candidates, split_list)
    for k, (_, test_idx) in enumerate(split_list, start=1):        # the regime splits ARE the model splits
        got = np.sort(predictions.loc[(predictions["fold"] == k) & (predictions["model"] == candidates[0].name), "row"].to_numpy())
        if not np.array_equal(got, np.asarray(test_idx)):
            raise Phase10CError(f"fold {k}: regime split does not match the walk-forward split")
    states, regime_folds = rg.assign_oos_states(_values(evaluable), split_list)
    unconditional = _subset_result(predictions, set(predictions["row"].unique().tolist()), candidates, config, 0)
    regimes = {}
    for fam, fd in rg.FAMILIES.items():
        regimes[fam] = {s: _subset_result(predictions, {r for r, st in states[fam].items() if st == s},
                                          candidates, config, min_rows) for s in fd["states"]}
    warn_counts = Counter(f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught)
    return {
        "harness_version": HARNESS_VERSION, "gate_version": GATE_VERSION, "significance_level": config.significance_level,
        "horizon_days": horizon, "n_splits": n_splits, "gap": config.gap, "features": list(feature_columns),
        "target": target_definition(horizon), "min_state_oos_rows": min_rows,
        "frame": {"rows": len(data), "first_date": data["Date"].iloc[0].date().isoformat(),
                  "last_date": data["Date"].iloc[-1].date().isoformat()},
        "training_period": {"start": folds[0]["train_start"], "end": folds[-1]["train_end"]},
        "evaluation_period": {"start": folds[0]["test_start"], "end": folds[-1]["test_end"],
                              "n_oos_rows": int(sum(f["n_test"] for f in folds))},
        "folds": folds, "fold_metrics": fold_metrics.to_dict(orient="records"),
        "regime_folds": regime_folds, "unconditional_reference": unconditional, "regimes": regimes,
        "warnings": {"total": len(caught), "by_message": dict(warn_counts)},
    }


# ==========================================
# Development
# ==========================================


def run_development(registry_path: Path, frame: pd.DataFrame, inputs: dict, *, spec: Spec10C = OFFICIAL,
                    results_dir: Path | None = None, evaluate=evaluate_regimes, log=print) -> dict:
    registry_path = Path(registry_path)
    reg = load_registry(registry_path)
    if reg["status"] != "FROZEN" or reg["development"] is not None:
        raise Phase10CError("development has already been run for this registry")
    matrix = _check_spec(reg, spec)
    _check_inputs(reg, inputs)
    results_dir = Path(results_dir) if results_dir is not None else registry_path.parent
    exp_dir = results_dir / "experiments"
    planned = [exp_dir / f"dev_{fs}_h{h}.json" for fs in spec.feature_sets for h in spec.horizons]
    if any(p.exists() for p in planned):
        raise Phase10CError(f"development artifacts already exist in {exp_dir}")
    dev = development_frame(frame, spec)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    comps, runs = [], []
    for fs, fdef in spec.feature_sets.items():
        cols = list(fdef["columns"])
        for h in spec.horizons:
            name = f"phase10c_dev_{fs}_h{h}"
            log(f"[develop] {name}: {len(cols)} features, horizon {h} ...")
            report = {"experiment": name, "phase": PHASE, "feature_set": fs, "matrix_sha256": reg["matrix_sha256"],
                      "inputs": inputs, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      **evaluate(dev, cols, spec.candidates(), horizon=h, n_splits=spec.n_splits,
                                 min_rows=spec.min_state_oos)}
            artifact = _write_new_json(exp_dir / f"dev_{fs}_h{h}.json", report)
            runs.append({"feature_set": fs, "horizon": h, "artifact": artifact,
                         "evaluation_period": report["evaluation_period"], "warnings_total": report["warnings"]["total"],
                         "unconditional_reference": {m["model"]: m["qualification"]["status"]
                                                     for m in report["unconditional_reference"]["models"]}})
            for fam, fd in rg.FAMILIES.items():
                for st in fd["states"]:
                    res = report["regimes"][fam][st]
                    for c in spec.candidates():
                        if c.is_baseline:
                            continue
                        cid = comparison_id(fs, h, c.name, fam, st)
                        entry = {"comparison_id": cid, "experiment_id": base_experiment_id(fs, h, c.name),
                                 "feature_set": fs, "horizon": h, "model": c.name, "task": c.task, "family": fam,
                                 "state": st, "n_oos": res["n_oos"], "artifact": artifact["path"]}
                        if res["status"] == INSUFFICIENT_DATA:
                            entry.update({"status": INSUFFICIENT_DATA, "eligible": False})
                        else:
                            mm = {m["model"]: m for m in res["models"]}
                            q = mm[c.name]["qualification"]
                            entry.update({"eligible": True, "gate_status": q["status"],
                                          "reference_baseline": q["reference_baseline"],
                                          "primary_metric": q["primary_metric"], "model_value": q["model_value"],
                                          "baseline_value": q["baseline_value"],
                                          "dm_stat": q["diebold_mariano"]["dm_stat"],
                                          "p_value_raw": q["diebold_mariano"]["p_value"], "reasons": q["reasons"],
                                          "accuracy": mm[c.name]["metrics"].get("Accuracy"),
                                          "always_up_accuracy": mm.get("Always UP", {}).get("metrics", {}).get("Accuracy")})
                        comps.append(entry)
    if sorted(c["comparison_id"] for c in comps) != sorted(matrix["comparisons"]):
        raise Phase10CError("results do not cover exactly the pre-registered comparisons")
    eligible = [c for c in comps if c["eligible"]]
    holm = holm_adjust([c["p_value_raw"] for c in eligible])
    for c, p in zip(eligible, holm):
        c["p_value_holm"] = p
        ok = (c["gate_status"] == QUALIFIED and c["n_oos"] >= spec.min_state_oos
              and c["p_value_raw"] < ALPHA and p < HOLM_ALPHA)
        c["status"] = "QUALIFIED" if ok else "NOT_QUALIFIED"
    qualifiers = [c["comparison_id"] for c in eligible if c["status"] == "QUALIFIED"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    reg["results"] = {c["comparison_id"]: c for c in comps}
    reg["development"] = {
        "status": "COMPLETE", "started_at": started, "finished_at": now, "git": git_info(),
        "environment": environment(), "inputs_verified": True,
        "frame": {"rows": len(dev), "first_date": dev["Date"].iloc[0].date().isoformat(),
                  "last_date": dev["Date"].iloc[-1].date().isoformat()},
        "runs": runs, "n_base_experiments": matrix["n_base_experiments"], "n_comparisons": len(comps),
        "n_insufficient": len(comps) - len(eligible), "family_size": len(eligible),
        "multiple_comparisons": {"method": "Holm", "family_size": len(eligible),
                                 "n_raw_p_below_alpha": sum(c["p_value_raw"] < ALPHA for c in eligible),
                                 "n_holm_p_below_alpha": sum(p < HOLM_ALPHA for p in holm),
                                 "n_gate_qualified_before_holm": sum(c["gate_status"] == QUALIFIED for c in eligible)},
        "warnings_total": sum(r["warnings_total"] for r in runs), "qualifiers": qualifiers,
        "result": (f"{len(qualifiers)} DEVELOPMENT QUALIFIER(S)" if qualifiers else NO_DEVELOPMENT_QUALIFIERS),
        "determinism": DETERMINISM_NOT_VERIFIED,
    }
    if qualifiers:
        reg["status"] = "DEVELOPMENT_COMPLETE"
    else:
        reg["holdout"] = {"status": "NOT_APPLICABLE", "result": NO_DEVELOPMENT_QUALIFIERS, "recorded_at": now,
                          "explanation": "no development qualifier; the Phase-10C confirmation holdout was not read",
                          "confirmed": []}
        reg["status"] = "COMPLETE"
    _write_json_atomic(registry_path, reg)
    return reg


# ==========================================
# Confirmation holdout (qualifiers + explicit authorization only)
# ==========================================


def run_holdout(registry_path: Path, frame: pd.DataFrame | None, inputs: dict | None, *, confirm: bool,
                spec: Spec10C = OFFICIAL, results_dir: Path | None = None, log=print) -> dict:
    if not confirm:
        raise Phase10CError("the Phase-10C confirmation holdout runs only with explicit authorization (--confirm)")
    registry_path = Path(registry_path)
    reg = load_registry(registry_path)
    if reg["status"] != "DEVELOPMENT_COMPLETE" or reg["holdout"] is not None:
        raise Phase10CError(f"holdout requires development qualifiers and runs exactly once (status {reg['status']})")
    _check_spec(reg, spec)
    qualifiers = list(reg["development"]["qualifiers"])
    if not qualifiers:
        raise Phase10CError("no development qualifiers; the holdout is not applicable")
    if frame is None or inputs is None:
        raise Phase10CError("development qualifiers exist; the research frame and inputs are required")
    _check_inputs(reg, inputs)
    results_dir = Path(results_dir) if results_dir is not None else registry_path.parent
    by_name = {c.name: c for c in spec.candidates()}
    done, confirmed = {}, []
    for cid in qualifiers:
        c = reg["results"][cid]
        key = (c["feature_set"], c["horizon"], c["model"])
        if key not in done:
            cand = by_name[c["model"]]
            cands = [x for x in spec.candidates() if x.task == cand.task and x.is_baseline] + [cand]
            h = c["horizon"]
            log(f"[holdout] {c['experiment_id']} ...")
            out = evaluate_regimes(frame, list(spec.feature_sets[c["feature_set"]]["columns"]), cands, horizon=h,
                                   n_splits=spec.n_splits, min_rows=spec.min_state_oos,
                                   splits=lambda ev, h=h: holdout_splits(ev["Date"], h, spec.n_splits,
                                                                         spec.holdout_start, spec.holdout_end))
            done[key] = (out, _write_new_json(results_dir / "experiments" / f"holdout_{c['experiment_id']}.json",
                                              {"experiment_id": c["experiment_id"], "inputs": inputs, **out}))
        out, artifact = done[key]
        res = out["regimes"][c["family"]][c["state"]]
        if res["status"] == INSUFFICIENT_DATA:
            c["holdout"] = {"status": INSUFFICIENT_DATA, "n_oos": res["n_oos"], "artifact": artifact}
        else:
            q = {m["model"]: m for m in res["models"]}[c["model"]]["qualification"]
            c["holdout"] = {"status": q["status"], "n_oos": res["n_oos"], "gate": q, "artifact": artifact}
            if q["status"] == QUALIFIED:
                confirmed.append(cid)
        c["status"] = "CONFIRMED" if cid in confirmed else "NOT_CONFIRMED"
    reg["holdout"] = {"status": "COMPLETE", "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
             f"Matrix   : {m['n_base_experiments']} base experiments = {len(m['feature_sets'])} feature sets x "
             f"{len(m['horizons'])} horizons x {sum(not c['is_baseline'] for c in m['candidates'])} models; "
             f"{m['regimes']['state_count']} regime states -> max {m['n_comparisons_max']} comparisons",
             "Features : " + ", ".join(f"{k} {v['n_features']}" for k, v in m["feature_sets"].items())]
    if reg.get("regime_diagnostics"):
        lines.append(f"Regimes  : expected eligible family size {reg['regime_diagnostics']['expected_family_size']}")
    if reg.get("development"):
        d = reg["development"]
        mc_ = d["multiple_comparisons"]
        lines.append(f"Develop  : {d['result']}  (family {d['family_size']}, insufficient {d['n_insufficient']}; "
                     f"gate-qualified before Holm {mc_['n_gate_qualified_before_holm']}; raw p<0.05 "
                     f"{mc_['n_raw_p_below_alpha']}; Holm p<0.05 {mc_['n_holm_p_below_alpha']})")
    if reg.get("holdout"):
        lines.append(f"Holdout  : {reg['holdout']['status']} - {reg['holdout']['result']}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 10C regime-conditioned research (pre-registered)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "regime-diagnostics", "develop", "holdout", "status"):
        p = sub.add_parser(name)
        p.add_argument("--registry", type=Path, default=REGISTRY_PATH)
        p.add_argument("--store", type=Path, default=DEFAULT_STORE)
        p.add_argument("--spy", type=Path, default=DEFAULT_SPY)
        p.add_argument("--qqq", type=Path, default=DEFAULT_QQQ)
        if name == "holdout":
            p.add_argument("--confirm", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "status":
        print("\n".join(_status_lines(load_registry(args.registry))))
        return 0
    if args.command == "holdout":
        if not args.confirm:
            raise Phase10CError("the Phase-10C confirmation holdout runs only with explicit authorization (--confirm)")
        reg = load_registry(args.registry)
        if reg["status"] != "DEVELOPMENT_COMPLETE" or not reg["development"]["qualifiers"]:
            raise Phase10CError(f"holdout not applicable (registry status {reg['status']})")
    inputs = collect_inputs(args.store, args.spy, args.qqq)
    frame, prov, aapl = build_frame(args.store, args.spy, args.qqq)
    if args.command == "freeze":
        rebuilt, _, _ = build_frame(args.store, args.spy, args.qqq)
        reg = freeze(args.registry, inputs=inputs, validation=freeze_validation(frame, prov, aapl, OFFICIAL, rebuilt))
        print(f"Frozen   : {args.registry}")
        print("\n".join(_status_lines(reg)))
        return 0
    if args.command == "regime-diagnostics":
        diag = run_regime_diagnostics(args.registry, frame, inputs)
        for h, d in diag["horizons"].items():
            print(f"h={h}: {d['oos_rows']} OOS rows; eligible states {d['n_eligible_states']}/{rg.STATE_COUNT}")
            for fam, cnt in d["pooled_oos_state_counts"].items():
                print(f"   {fam:20s} " + "  ".join(f"{s}={n}" for s, n in cnt.items()))
        print(f"expected eligible family size: {diag['expected_family_size']}")
        return 0
    if args.command == "develop":
        print("\n".join(_status_lines(run_development(args.registry, frame, inputs))))
        return 0
    print("\n".join(_status_lines(run_holdout(args.registry, frame, inputs, confirm=True))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
