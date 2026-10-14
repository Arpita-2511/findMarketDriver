"""
Phase 11B Stage 11B.4: development research runner (feature sets B, C, D, F).

Frozen protocol (design_freeze.json; Phase 11A code reused unchanged):
    target      excess return over SPY; direction = 1 if above 0; horizons 1, 3, 5
    models      Phase 9 candidates (6 regression + 5 classification), fixed parameters, seed 42, no tuning
    per stock   training.phase10a_research.evaluate_excess (TimeSeriesSplit 20, gap = h)
    panel       training.phase11a_research.evaluate_panel (date-grouped folds, symbol never a feature,
                per-stock baselines from training rows, DM on per-date cross-sectional mean loss)
    counts      exactly 3828 per-stock / 132 panel / 132 panel-without-AAPL (descriptive)
    11B.4       records metrics, gate outcomes, raw p-values and DESCRIPTIVE 11B-stage Holm only;
                final qualification is decided in 11B.5 over the frozen 5,742 / 198 families
    confirmation period: NOT_EVALUATED; never read here

Pre-development steps, in this order (each writes a write-once evidence file under
data/results/research/phase11/11B/prerequisites/; each requires every earlier step):

    python -m training.phase11b_research validate --what leakage   # 1. T1-T15 (pytest, nothing deselected)
    python -m training.phase11b_research validate --what smoke     # 2. synthetic smoke test (synthetic data)
    python -m training.phase11b_research validate --what aapl      # 3. real AAPL per-stock validation (B/C/D/F, h=1)
    python -m training.phase11b_research validate --what stock     # 4. real non-AAPL per-stock validation (AXP)
    python -m training.phase11b_research validate --what panel     # 5. real panel validation (F, h=1, 29 stocks)
    python -m training.phase11b_research freeze                    # 6. registry freeze (write-once)
    python -m training.phase11b_research develop [--workers 12]    # the 3,960-comparison matrix
    python -m training.phase11b_research status

Validation evidence records structure only (rows, folds, out-of-sample rows, model coverage,
finite metrics) - never metrics, gate outcomes or p-values.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import training.phase10a_research as p10a  # noqa: E402
import training.phase11_design as p11d  # noqa: E402
import training.phase11a_research as a11  # noqa: E402
import training.phase9_research as p9  # noqa: E402
from services.finbert_sentiment import build_sentiment_text, text_hash  # noqa: E402
from training.build_dataset import file_sha256, meta_path_for  # noqa: E402
from training.evaluation_harness import (  # noqa: E402
    GATE_VERSION,
    HARNESS_VERSION,
    QUALIFIED,
    REGRESSION,
    git_info,
)
from training.excess_targets import SPY_CLOSE, target_definition  # noqa: E402
from training.phase11b_feature_build import FRAME_COLUMNS  # noqa: E402
from training.phase11b_sentiment_scoring import (  # noqa: E402
    PHASE_11B_DIR,
    REPORT_PATH,
    load_integrity_report,
    partition_canonical,
    verify_frozen_artifacts,
    verify_partition_against_report,
)
from training.phase9_research import (  # noqa: E402
    ResearchError,
    _slug,
    _write_new_json,
    canonical_json,
    environment,
    holm_adjust,
)

PHASE = "11B"
STAGE = "11B.4"
REGISTRY_VERSION = "phase11b_registry_v1"
STAGE_FEATURE_SETS = ("B", "C", "D", "F")
FEATURE_SETS = {k: p11d.FEATURE_SETS[k] for k in STAGE_FEATURE_SETS}
HORIZONS = (1, 3, 5)
ALPHA = a11.ALPHA
HOLM_ALPHA = a11.HOLM_ALPHA
DEFERRED = "DEFERRED_TO_11B_5"

# ---- pins (each anchored by the commit named) ----
FEATURE_VERIFICATION_PATH = PHASE_11B_DIR / "feature_verification.json"
FEATURE_VERIFICATION_SHA256 = "b613f1277eed3656abe53146126cedf29dec0215b34c91c626d488de646eaf19"   # 9adb240
PHASE_11A_REGISTRY_SHA256 = "d5cdef22a634a98133e9a3643fd7d1590fbf2e712634550473d3be8699f6ae5a"     # e6a50f5
CODE_FILES = ("training/phase11b_research.py", "training/phase11a_research.py", "training/phase10a_research.py",
              "training/phase9_research.py", "training/evaluation_harness.py", "training/excess_targets.py")
LEAKAGE_TEST_FILE = "tests/unit/test_phase11b_leakage.py"
STEPS = ("leakage", "smoke", "aapl", "stock", "panel")
REAL_STEPS = ("aapl", "stock", "panel")


class Phase11BError(ResearchError):
    """Phase 11B.4 protocol violation; the run stops."""


@dataclass(frozen=True)
class Spec:
    """Frozen expectations of the real run; any other value is a synthetic fixture and is labelled as such."""
    universe: tuple = tuple(p11d.UNIVERSE)
    rows_per_stock: int = 1928
    oos_rows_per_stock_h1: int = 1820
    panel_oos_rows_h1: int = 52780
    n_splits: int = a11.N_SPLITS
    expected_counts: tuple = (3828, 132, 132)
    dev_start: date = date(2017, 2, 1)
    dev_end: date = date(2024, 9, 30)
    feature_verification_path: Path = FEATURE_VERIFICATION_PATH
    feature_verification_sha256: str = FEATURE_VERIFICATION_SHA256


FROZEN_SPEC = Spec()


def is_real(spec: Spec) -> bool:
    return spec == FROZEN_SPEC


def spec_sha256(spec: Spec) -> str:
    d = {k: (str(v) if isinstance(v, (Path, date)) else v) for k, v in dataclasses.asdict(spec).items()}
    d["feature_verification_path"] = Path(spec.feature_verification_path).name
    return hashlib.sha256(canonical_json(d).encode("utf-8")).hexdigest()


def code_sha256() -> dict:
    return {f: file_sha256(PROJECT_ROOT / f) for f in CODE_FILES}


@dataclass(frozen=True)
class Paths:
    out_dir: Path = PHASE_11B_DIR

    @property
    def registry(self) -> Path:
        return self.out_dir / "registry.json"

    @property
    def prerequisites(self) -> Path:
        return self.out_dir / "prerequisites"

    @property
    def staging(self) -> Path:
        return self.out_dir / "development.incomplete"

    @property
    def development(self) -> Path:
        return self.out_dir / "development"

    @property
    def results(self) -> Path:
        return self.out_dir / "development_results.json"

    def evidence(self, step: str) -> Path:
        return self.prerequisites / f"{STEPS.index(step) + 1:02d}_{step}.json"


DEFAULT_PATHS = Paths()


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ============================================================
# Verified inputs and frames
# ============================================================


def verified_inputs(spec: Spec = FROZEN_SPEC) -> dict:
    """Every frozen and upstream hash; the 11B.3 verification record and every feature frame it names."""
    frozen = verify_frozen_artifacts()                                       # 11B design freeze, Amendments 03-05
    if p11d.load_registry()["design_sha256"] != a11.FROZEN_DESIGN_SHA:
        raise Phase11BError("frozen Phase 11.0 design registry changed")
    for path, pinned, name in ((a11.AMENDMENT, a11.AMENDMENT_SHA, "amendment_01"),
                               (a11.PRICE_REPORT, a11.PRICE_REPORT_SHA, "Phase 11.1 price verification"),
                               (a11.REGISTRY_PATH, PHASE_11A_REGISTRY_SHA256, "Phase 11A registry")):
        if file_sha256(path) != pinned:
            raise Phase11BError(f"{name} changed (stop condition: 11A artifacts must be unmodified)")
    vpath = Path(spec.feature_verification_path)
    if not vpath.is_file() or file_sha256(vpath) != spec.feature_verification_sha256:
        raise Phase11BError("feature_verification.json is missing or differs from its pinned SHA256")
    v = json.loads(vpath.read_text(encoding="utf-8"))
    required_true = ("shared_columns_match_11a", "f_duplicate_check_passed", "prefix_invariance_check_passed",
                     "horizon_independence_check_passed", "phase_11a_p_values_hash_verified",
                     "news_file_hashes_match_records", "sentiment_file_hashes_match_records",
                     "event_file_hashes_match_records", "partition_reproduces_integrity_report")
    checks = {
        "document_type": v.get("document_type") == "phase_11b_feature_verification",
        "stage": v.get("stage") == "11B.3",
        **{k: v.get(k) is True for k in required_true},
        "nans_in_development_features": v.get("nans_in_development_features") == 0,
        "row_count_per_stock": v.get("row_count_per_stock") == spec.rows_per_stock,
        "panel_oos_rows_all29_h1": v.get("panel_oos_rows_all29_h1") == spec.panel_oos_rows_h1,
        "final_families": (v.get("final_panel_family_size"), v.get("final_per_stock_family_size")) == (198, 5742),
        "sealed_content_used": v.get("sealed_content_used") is False,
        "phase_8_score_reuse": v.get("phase_8_score_reuse") is False,
        "feature_sets": all(v.get("feature_sets", {}).get(k) == list(FEATURE_SETS[k]["columns"]) for k in STAGE_FEATURE_SETS),
        "frame_columns": v.get("frame_columns") == list(FRAME_COLUMNS),
        "universe": sorted(v.get("frame_files", {})) == sorted(spec.universe),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise Phase11BError(f"feature_verification.json fails 11B.4 input checks: {failed}")
    frames = {}
    for sym in sorted(spec.universe):
        info = v["frame_files"][sym]
        if any(v["frame_paths"][fs][sym] != info["path"] for fs in STAGE_FEATURE_SETS):
            raise Phase11BError(f"{sym}: frame_paths disagree with frame_files")
        if file_sha256(_resolve(info["path"])) != info["sha256"]:
            raise Phase11BError(f"{sym}: feature frame differs from its recorded SHA256")
        frames[sym] = {"path": info["path"], "sha256": info["sha256"]}
    return {"frozen_artifact_sha256": frozen, "design_sha256": a11.FROZEN_DESIGN_SHA,
            "phase11a_registry_sha256": PHASE_11A_REGISTRY_SHA256,
            "feature_verification": {"file": vpath.name, "sha256": spec.feature_verification_sha256},
            "feature_run_id": v["run_id"], "frames": frames}


def inputs_sha256(inputs: dict) -> str:
    return hashlib.sha256(canonical_json(inputs).encode("utf-8")).hexdigest()


def load_frames(inputs: dict, spec: Spec = FROZEN_SPEC) -> dict[str, pd.DataFrame]:
    """Hash-verified development frames only; any date outside the development period stops the run."""
    frames, dates = {}, None
    feature_cols = sorted({c for fs in FEATURE_SETS.values() for c in fs["columns"]})
    for sym, info in sorted(inputs["frames"].items()):
        path = _resolve(info["path"])
        if file_sha256(path) != info["sha256"]:
            raise Phase11BError(f"{sym}: feature frame changed after verification")
        f = pd.read_csv(path, float_precision="round_trip")
        if list(f.columns) != list(FRAME_COLUMNS):
            raise Phase11BError(f"{sym}: frame columns differ from the frozen layout")
        if len(f) != spec.rows_per_stock:
            raise Phase11BError(f"{sym}: {len(f)} rows, expected {spec.rows_per_stock}")
        f["Date"] = pd.to_datetime(f["Date"], format="%Y-%m-%d").astype("datetime64[ns]")
        d = [x.date() for x in f["Date"]]
        if d != sorted(set(d)):
            raise Phase11BError(f"{sym}: dates are not strictly increasing")
        if d[0] < spec.dev_start or d[-1] > spec.dev_end:
            raise Phase11BError(f"{sym}: dates outside the development period "
                                f"({d[0]} .. {d[-1]}); the confirmation period is NOT_EVALUATED")
        if is_real(spec) and (d[0], d[-1]) != (spec.dev_start, spec.dev_end):
            raise Phase11BError(f"{sym}: development sessions do not span {spec.dev_start} .. {spec.dev_end}")
        if dates is not None and d != dates:
            raise Phase11BError("stocks do not share the same development sessions")
        dates = d
        if not np.isfinite(f[feature_cols + ["Close", SPY_CLOSE]].to_numpy(dtype="float64")).all():
            raise Phase11BError(f"{sym}: non-finite development values")
        frames[sym] = f
    return frames


# ============================================================
# Matrix
# ============================================================


def matrix(spec: Spec = FROZEN_SPEC) -> dict:
    d = p11d.design()
    if p11d.STAGES["11B"]["feature_sets"] != list(STAGE_FEATURE_SETS):
        raise Phase11BError("11B feature sets differ from the frozen staging")
    models = [c for c in a11.candidates() if not c.is_baseline]
    per_stock = [p11d.per_stock_id(s, f, h, c.name) for s in spec.universe for f in STAGE_FEATURE_SETS
                 for h in HORIZONS for c in models]
    panel = [p11d.panel_id(f, h, c.name) for f in STAGE_FEATURE_SETS for h in HORIZONS for c in models]
    sens = [p11d.panel_id(f, h, c.name, True) for f in STAGE_FEATURE_SETS for h in HORIZONS for c in models]
    for mine, key in ((per_stock, "per_stock"), (panel, "panel"), (sens, "aapl_excluded_sensitivity")):
        if not set(mine) <= set(d["comparison_ids"][key]) or len(set(mine)) != len(mine):
            raise Phase11BError(f"11B {key} ids are not a subset of the frozen design")
    counts = (len(per_stock), len(panel), len(sens))
    if counts != tuple(spec.expected_counts) or (is_real(spec) and counts != (3828, 132, 132)):
        raise Phase11BError(f"comparison counts {counts} differ from the frozen {spec.expected_counts}")
    design = json.loads((PHASE_11B_DIR / "design_freeze.json").read_text(encoding="utf-8"))
    return {
        "phase": PHASE, "stage": "11B.4 (news feature sets B, C, D, F)",
        "design_sha256": a11.FROZEN_DESIGN_SHA, "universe": list(spec.universe),
        "feature_sets": FEATURE_SETS, "horizons": list(HORIZONS),
        "targets": {str(h): target_definition(h) for h in HORIZONS},
        "models": [{"name": c.name, "task": c.task} for c in models],
        "baselines": {"per_stock": ["Mean Return", "Zero Return", "Always UP", "Base Rate"],
                      "panel": ["Mean Return", "Zero Return", a11.PER_STOCK_MEAN.name, "Always UP", "Base Rate",
                                a11.PER_STOCK_RATE.name]},
        "evaluation": {"harness_version": HARNESS_VERSION, "gate_version": GATE_VERSION, "n_splits": spec.n_splits,
                       "gap": "horizon", "significance_level": ALPHA,
                       "per_stock": "training.phase10a_research.evaluate_excess (unchanged)",
                       "panel": "training.phase11a_research.evaluate_panel (unchanged)",
                       "development": [spec.dev_start.isoformat(), spec.dev_end.isoformat()],
                       "confirmation": "NOT_EVALUATED"},
        "families": {"stage_11b_descriptive": {"panel": len(panel), "per_stock": len(per_stock),
                                               "method": "Holm", "use": "descriptive only"},
                     "final": design["multiple_testing"]["final_families"],
                     "aapl_excluded_sensitivity": {"size": len(sens), "role": "descriptive only, never qualifies"}},
        "qualification": {"in_11b_4": "none - metrics, gate outcomes, raw p-values, descriptive stage Holm",
                          "final": "11B.5: gate QUALIFIED AND raw p < 0.05 AND Holm p < 0.05 within the final families",
                          "status": DEFERRED},
        "amendment_02": ("descriptive only: AAPL B/C/D/F vs Phase 10A results, and development-only article "
                         "overlap / text hash / availability vs the Phase 8 AAPL dataset"),
        "comparison_ids": {"per_stock": per_stock, "panel": panel, "aapl_excluded_sensitivity": sens},
        "counts": {"per_stock": len(per_stock), "panel": len(panel), "aapl_excluded_sensitivity": len(sens)},
    }


def matrix_sha256(m: dict) -> str:
    return hashlib.sha256(canonical_json(m).encode("utf-8")).hexdigest()


# ============================================================
# Pre-development evidence
# ============================================================


def _evidence_context(spec: Spec, inputs: dict | None) -> dict:
    return {"spec_sha256": spec_sha256(spec), "inputs_sha256": inputs_sha256(inputs) if inputs else None,
            "code_sha256": code_sha256()}


def evidence_status(step: str, spec: Spec, inputs: dict | None, paths: Paths) -> tuple[bool, str]:
    path = paths.evidence(step)
    if not path.is_file():
        return False, "missing"
    ev = json.loads(path.read_text(encoding="utf-8"))
    ctx = _evidence_context(spec, inputs)
    if ev.get("step") != step or ev.get("outcome") != "PASS":
        return False, "not a PASS for this step"
    if ev.get("spec_sha256") != ctx["spec_sha256"]:
        return False, "recorded for a different specification"
    if step != "leakage" and step != "smoke" and ev.get("inputs_sha256") != ctx["inputs_sha256"]:
        return False, "recorded for different inputs"
    if ev.get("code_sha256") != ctx["code_sha256"]:
        return False, "recorded for different code"
    if step in REAL_STEPS and ev.get("data_kind") != ("real" if is_real(spec) else "synthetic_fixture"):
        return False, "data kind does not match the specification"
    return True, "valid"


def require_steps(upto: str, spec: Spec, inputs: dict | None, paths: Paths) -> None:
    for step in STEPS[:STEPS.index(upto)]:
        ok, why = evidence_status(step, spec, inputs, paths)
        if not ok:
            raise Phase11BError(f"pre-development step '{step}' is not complete ({why}); run it first, in order")


def _write_evidence(step: str, spec: Spec, inputs: dict | None, paths: Paths, data_kind: str, details: dict) -> dict:
    path = paths.evidence(step)
    if path.exists():
        raise Phase11BError(f"{path.name} already exists; evidence is write-once (move it aside manually to redo)")
    path.parent.mkdir(parents=True, exist_ok=True)
    ev = {"step": step, "order": STEPS.index(step) + 1, "stage": STAGE, "outcome": "PASS", "data_kind": data_kind,
          **_evidence_context(spec, inputs), "details": details,
          "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_info()}
    _write_new_json(path, ev)
    return ev


def run_leakage_tests() -> tuple[int, str]:
    proc = subprocess.run([sys.executable, "-m", "pytest", LEAKAGE_TEST_FILE, "-m",
                           "gate_authorization or not gate_authorization", "-q", "-p", "no:cacheprovider"],
                          cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=1800)
    return proc.returncode, proc.stdout + proc.stderr


def parse_pytest_summary(output: str) -> dict:
    tail = output.strip().splitlines()[-1] if output.strip() else ""
    counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|errors?|skipped|deselected|xfailed|xpassed)", tail)}
    return {"summary_line": tail, **counts}


def validate_leakage(spec: Spec = FROZEN_SPEC, paths: Paths = DEFAULT_PATHS,
                     runner: Callable[[], tuple[int, str]] = run_leakage_tests) -> dict:
    rc, out = runner()
    s = parse_pytest_summary(out)
    bad = {k: s.get(k, 0) for k in ("failed", "error", "errors", "skipped", "deselected", "xfailed", "xpassed")}
    if rc != 0 or s.get("passed", 0) == 0 or any(bad.values()):
        raise Phase11BError(f"T1-T15 leakage tests did not all pass with nothing skipped or deselected: {s}")
    return _write_evidence("leakage", spec, None, paths, "repository tests",
                           {"test_file": LEAKAGE_TEST_FILE, "rc": rc, **s})


def synthetic_frames(n_stocks: int = 3, n_rows: int = 160, seed: int = 11) -> dict[str, pd.DataFrame]:
    """Synthetic 11B-shaped frames (fabricated values only)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_rows)
    frames = {}
    for i in range(n_stocks):
        f = pd.DataFrame({"Date": dates.astype("datetime64[ns]"),
                          "Close": 50 + i + np.cumsum(rng.normal(0, 0.5, n_rows)) + 30,
                          SPY_CLOSE: 300 + np.cumsum(rng.normal(0, 0.5, n_rows))})
        for c in FRAME_COLUMNS[3:]:
            f[c] = rng.normal(0, 1, n_rows)
        frames[f"SYN{i}"] = f
    return frames


def validate_smoke(spec: Spec = FROZEN_SPEC, paths: Paths = DEFAULT_PATHS) -> dict:
    """Synthetic end-to-end exercise of the per-stock and panel code paths. Never real data."""
    require_steps("smoke", spec, None, paths)
    frames = synthetic_frames()
    cols = list(FEATURE_SETS["F"]["columns"])
    rep = p10a.evaluate_excess(frames["SYN0"], cols, a11.candidates(), horizon=1, n_splits=3)
    pan = a11.evaluate_panel(frames, cols, a11.panel_candidates(), horizon=3, n_splits=3)
    names = {m["model"] for m in rep["models"]}
    if names != {c.name for c in a11.candidates()} or len(rep["folds"]) != 3:
        raise Phase11BError("synthetic smoke test: per-stock run is incomplete")
    if pan["n_symbols"] != 3 or len(pan["folds"]) != 3 or \
            {m["model"] for m in pan["models"]} != {c.name for c in a11.panel_candidates()}:
        raise Phase11BError("synthetic smoke test: panel run is incomplete")
    return _write_evidence("smoke", spec, None, paths, "synthetic",
                           {"per_stock": {"feature_set": "F", "horizon": 1, "folds": len(rep["folds"]),
                                          "n_oos_rows": rep["evaluation_period"]["n_oos_rows"],
                                          "models": len(rep["models"])},
                            "panel": {"feature_set": "F", "horizon": 3, "folds": len(pan["folds"]),
                                      "n_symbols": pan["n_symbols"], "n_oos_rows": pan["n_oos_rows"],
                                      "models": len(pan["models"])},
                            "note": "fabricated data; exercises code paths only; not a real-data validation"})


def _check_stock_run(rep: dict, spec: Spec) -> dict:
    primary = {m["model"]: m["metrics"].get("MSE" if m["task"] == REGRESSION else "Log_Loss")
               for m in rep["models"] if not m["is_baseline"]}
    problems = []
    if rep["frame"]["rows"] != spec.rows_per_stock:
        problems.append("rows")
    if len(rep["folds"]) != spec.n_splits:
        problems.append("folds")
    if rep["horizon_days"] == 1 and rep["evaluation_period"]["n_oos_rows"] != spec.oos_rows_per_stock_h1:
        problems.append("n_oos_rows")
    if {m["model"] for m in rep["models"]} != {c.name for c in a11.candidates()}:
        problems.append("models")
    if not all(v is not None and np.isfinite(v) for v in primary.values()):
        problems.append("non-finite metric")
    if problems:
        raise Phase11BError(f"real per-stock validation failed: {problems}")
    return {"rows": rep["frame"]["rows"], "first_date": rep["frame"]["first_date"], "last_date": rep["frame"]["last_date"],
            "folds": len(rep["folds"]), "n_oos_rows": rep["evaluation_period"]["n_oos_rows"],
            "models": len(rep["models"]), "warnings": rep["warnings"]["total"]}


def validate_stock(which: str, spec: Spec = FROZEN_SPEC, paths: Paths = DEFAULT_PATHS) -> dict:
    """Real per-stock validation: B, C, D, F at h=1 for AAPL ('aapl') or the first non-AAPL stock ('stock')."""
    inputs = verified_inputs(spec)
    require_steps(which, spec, inputs, paths)
    frames = load_frames(inputs, spec)
    sym = "AAPL" if which == "aapl" else next(s for s in spec.universe if s != "AAPL")
    details = {"symbol": sym, "horizon": 1, "runs": {}}
    for fs in STAGE_FEATURE_SETS:
        rep = p10a.evaluate_excess(frames[sym], list(FEATURE_SETS[fs]["columns"]), a11.candidates(), horizon=1,
                                   n_splits=spec.n_splits)
        details["runs"][fs] = _check_stock_run(rep, spec)
    return _write_evidence(which, spec, inputs, paths, "real" if is_real(spec) else "synthetic_fixture", details)


def validate_panel(spec: Spec = FROZEN_SPEC, paths: Paths = DEFAULT_PATHS) -> dict:
    """Real panel validation: F at h=1 over the whole universe."""
    inputs = verified_inputs(spec)
    require_steps("panel", spec, inputs, paths)
    frames = load_frames(inputs, spec)
    r = a11.evaluate_panel(frames, list(FEATURE_SETS["F"]["columns"]), a11.panel_candidates(), horizon=1,
                           n_splits=spec.n_splits)
    problems = []
    if r["n_symbols"] != len(spec.universe):
        problems.append("n_symbols")
    if r["n_oos_rows"] != spec.panel_oos_rows_h1:
        problems.append("n_oos_rows")
    if len(r["folds"]) != spec.n_splits:
        problems.append("folds")
    if {m["model"] for m in r["models"]} != {c.name for c in a11.panel_candidates()}:
        problems.append("models")
    if problems:
        raise Phase11BError(f"real panel validation failed: {problems}")
    return _write_evidence("panel", spec, inputs, paths, "real" if is_real(spec) else "synthetic_fixture",
                           {"feature_set": "F", "horizon": 1, "n_symbols": r["n_symbols"], "n_rows": r["n_rows"],
                            "n_dates": r["n_dates"], "n_oos_rows": r["n_oos_rows"], "folds": len(r["folds"]),
                            "oos_period": r["oos_period"], "models": len(r["models"]),
                            "warnings": r["warnings"]["total"]})


# ============================================================
# Registry
# ============================================================


def freeze(spec: Spec = FROZEN_SPEC, paths: Paths = DEFAULT_PATHS) -> dict:
    if paths.registry.exists():
        raise Phase11BError(f"{paths.registry.name} already exists; the 11B registry is frozen once")
    inputs = verified_inputs(spec)
    require_steps("panel", spec, inputs, paths)
    ok, why = evidence_status("panel", spec, inputs, paths)
    if not ok:
        raise Phase11BError(f"pre-development step 'panel' is not complete ({why})")
    m = matrix(spec)
    reg = {"registry_version": REGISTRY_VERSION, "phase": PHASE, "stage": STAGE, "matrix": m,
           "matrix_sha256": matrix_sha256(m), "inputs": inputs, "inputs_sha256": inputs_sha256(inputs),
           "spec_sha256": spec_sha256(spec), "data_kind": "real" if is_real(spec) else "synthetic_fixture",
           "code_sha256": code_sha256(),
           "prerequisites": {s: {"file": paths.evidence(s).name, "sha256": file_sha256(paths.evidence(s))} for s in STEPS},
           "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "frozen_git": git_info(),
           "environment": environment(), "status": "FROZEN", "confirmation": "NOT_EVALUATED"}
    _write_new_json(paths.registry, reg)
    load_registry(spec, paths)                                              # verify what was written
    return reg


def load_registry(spec: Spec = FROZEN_SPEC, paths: Paths = DEFAULT_PATHS) -> dict:
    if not paths.registry.is_file():
        raise Phase11BError(f"{paths.registry.name} not found (run `freeze` first)")
    reg = json.loads(paths.registry.read_text(encoding="utf-8"))
    if reg.get("registry_version") != REGISTRY_VERSION or reg.get("status") != "FROZEN":
        raise Phase11BError("11B registry has an unexpected version or status")
    if matrix_sha256(reg["matrix"]) != reg["matrix_sha256"] or matrix_sha256(matrix(spec)) != reg["matrix_sha256"]:
        raise Phase11BError("11B registry matrix was edited or differs from the frozen matrix in code")
    if inputs_sha256(reg["inputs"]) != reg["inputs_sha256"] or reg["spec_sha256"] != spec_sha256(spec):
        raise Phase11BError("11B registry inputs or specification were edited")
    for s in STEPS:
        rec = reg["prerequisites"].get(s) or {}
        p = paths.evidence(s)
        if not p.is_file() or file_sha256(p) != rec.get("sha256"):
            raise Phase11BError(f"prerequisite evidence '{s}' changed after the registry was frozen")
    return reg


# ============================================================
# Development
# ============================================================


def _stock_job(sym: str, frame: pd.DataFrame, n_splits: int) -> dict:
    return {"symbol": sym, "runs": {f"{fs}_h{h}": p10a.evaluate_excess(frame, list(FEATURE_SETS[fs]["columns"]),
                                                                         a11.candidates(), horizon=h, n_splits=n_splits)
                                    for fs in STAGE_FEATURE_SETS for h in HORIZONS}}


def _panel_job(kind: str, fs: str, h: int, frames: dict[str, pd.DataFrame], n_splits: int) -> dict:
    use = {s: f for s, f in frames.items() if kind == "panel" or s != "AAPL"}
    return {"kind": kind, "fs": fs, "h": h,
            "result": a11.evaluate_panel(use, list(FEATURE_SETS[fs]["columns"]), a11.panel_candidates(),
                                         horizon=h, n_splits=n_splits)}


def _run_jobs(frames, spec: Spec, workers: int, log) -> tuple[dict, dict]:
    stock_res, panel_res = {}, {}
    panel_args = [(k, fs, h) for k in ("panel", "panel_exAAPL") for fs in STAGE_FEATURE_SETS for h in HORIZONS]
    if workers <= 1:
        for s in spec.universe:
            stock_res[s] = _stock_job(s, frames[s], spec.n_splits)["runs"]
        for k, fs, h in panel_args:
            panel_res[(k, fs, h)] = _panel_job(k, fs, h, frames, spec.n_splits)["result"]
        return stock_res, panel_res
    with ProcessPoolExecutor(max_workers=workers, initializer=a11._init_worker) as pool:
        futs = {pool.submit(_panel_job, k, fs, h, frames, spec.n_splits): ("panel", k, fs, h) for k, fs, h in panel_args}
        futs.update({pool.submit(_stock_job, s, frames[s], spec.n_splits): ("stock", s) for s in spec.universe})
        for f in futs:
            key, r = futs[f], f.result()
            if key[0] == "stock":
                stock_res[key[1]] = r["runs"]
                log(f"[11B] stock {key[1]} done")
            else:
                panel_res[key[1:]] = r["result"]
                log(f"[11B] {key[1]} {key[2]} h{key[3]} done")
    return stock_res, panel_res


def _entries(stock_res, panel_res, spec: Spec, artifacts: dict) -> tuple[list, list, list]:
    per_stock = []
    for s in spec.universe:
        for fs in STAGE_FEATURE_SETS:
            for h in HORIZONS:
                rep = stock_res[s][f"{fs}_h{h}"]
                base = {m["model"]: m["metrics"] for m in rep["models"] if m["is_baseline"]}
                for m in rep["models"]:
                    if m["is_baseline"]:
                        continue
                    q = m["qualification"]
                    per_stock.append({
                        "id": p11d.per_stock_id(s, fs, h, m["model"]), "symbol": s, "feature_set": fs, "horizon": h,
                        "model": m["model"], "task": m["task"], "n_oos": rep["evaluation_period"]["n_oos_rows"],
                        "n_folds": len(rep["folds"]), "metrics": m["metrics"],
                        "baseline_metrics": {k: v for k, v in base.items()
                                             if (k in ("Mean Return", "Zero Return")) == (m["task"] == REGRESSION)},
                        "gate": q, "gate_status": q["status"], "dm_stat": q["diebold_mariano"]["dm_stat"],
                        "loss_diff_mean": q["diebold_mariano"]["mean_loss_improvement"],
                        "p_value_raw": q["diebold_mariano"]["p_value"], "artifact": artifacts[f"stock_{s}"]["path"]})
    panel, sens = [], []
    for (kind, fs, h), r in sorted(panel_res.items()):
        base = {m["model"]: m["metrics"] for m in r["models"] if m["is_baseline"]}
        for m in r["models"]:
            if m["is_baseline"]:
                continue
            q = m["qualification"]
            e = {"id": p11d.panel_id(fs, h, m["model"], kind != "panel"), "feature_set": fs, "horizon": h,
                 "model": m["model"], "task": m["task"], "n_oos": r["n_oos_rows"], "n_dates": q["n_dates"],
                 "n_symbols": r["n_symbols"], "n_folds": len(r["folds"]), "metrics": m["metrics"],
                 "baseline_metrics": {k: v for k, v in base.items() if ("Return" in k) == (m["task"] == REGRESSION)},
                 "gate": q, "gate_status": q["status"], "dm_stat": q["diebold_mariano"]["dm_stat"],
                 "loss_diff_mean": q["diebold_mariano"]["mean_loss_improvement"],
                 "p_value_raw": q["diebold_mariano"]["p_value"], "artifact": artifacts[f"{kind}_{fs}_h{h}"]["path"]}
            (panel if kind == "panel" else sens).append(e)
    return per_stock, panel, sens


def stage_holm_descriptive(entries: list[dict], size: int) -> None:
    """Holm within the 11B stage only - DESCRIPTIVE. No entry is qualified here."""
    if len(entries) != size:
        raise Phase11BError(f"stage family has {len(entries)} comparisons, expected {size}")
    for e, p in zip(entries, holm_adjust([e["p_value_raw"] for e in entries])):
        e["p_value_holm_11b_stage_descriptive"] = p
        e["final_qualification"] = DEFERRED


def amendment02_results_comparison(per_stock: list[dict]) -> dict:
    """AAPL B/C/D/F vs Phase 10A development results; differences and gate-status agreement. Descriptive only."""
    r10a = json.loads(p10a.REGISTRY_PATH.read_text(encoding="utf-8"))["results"]
    comps = {}
    for e in per_stock:
        if e["symbol"] != "AAPL":
            continue
        d = r10a[p10a.experiment_id(e["feature_set"], e["horizon"], e["model"])]["development"]
        p11v, p10v = e["gate"]["model_value"], d["gate"]["model_value"]
        comps[f"{e['feature_set']}_h{e['horizon']}_{_slug(e['model'])}"] = {
            "status_11b": e["gate_status"], "status_10a": d["status"], "status_agree": e["gate_status"] == d["status"],
            "primary_11b": p11v, "primary_10a": p10v,
            "rel_primary_diff": abs(p11v / p10v - 1) if p10v else None,
            "p_11b": e["p_value_raw"], "p_10a": d["p_value_raw"], "abs_p_diff": abs(e["p_value_raw"] - d["p_value_raw"])}
    if len(comps) != 132:
        raise Phase11BError(f"Amendment 02 needs 132 AAPL comparisons, got {len(comps)}")
    return {"n_comparisons": len(comps), "n_status_agree": sum(c["status_agree"] for c in comps.values()),
            "max_rel_primary_diff": max(c["rel_primary_diff"] for c in comps.values() if c["rel_primary_diff"] is not None),
            "max_abs_p_diff": max(c["abs_p_diff"] for c in comps.values()), "comparisons": comps,
            "pass_fail": False, "affects_results": False}


def phase8_canonical() -> tuple[Path, str]:
    prov = p9.input_provenance(p10a.DEFAULT_STORE)["canonical_news"]          # hash-verified Phase 8 chain
    return _resolve(prov["path"]), prov["sha256"]


def amendment02_data_comparison(report_path: Path = REPORT_PATH,
                                phase8: Callable[[], tuple[Path, str]] = phase8_canonical) -> dict:
    """
    Development articles only on BOTH sides (information_available_at <= cutoff, frozen partition):
    AAPL-tagged 11B development articles vs Phase 8 AAPL development articles. Sealed records are
    kept as keys only by partition_canonical and enter no statistic. Descriptive only.
    """
    report = load_integrity_report(report_path)
    canonical = _resolve(report["canonical_input"]["path"])
    p11 = partition_canonical(canonical, expected_sha256=report["canonical_input"]["sha256"])
    verify_partition_against_report(p11, report)
    dev11 = {a.key: a for a in p11.development if "AAPL" in a.symbols}
    del p11
    p8_path, p8_sha = phase8()
    meta = json.loads(meta_path_for(p8_path).read_text(encoding="utf-8"))
    if meta.get("sha256") != p8_sha:
        raise Phase11BError("Phase 8 AAPL dataset metadata does not match its verified hash")
    dev8 = {a.key: a for a in partition_canonical(p8_path, expected_sha256=p8_sha).development}
    shared = sorted(set(dev11) & set(dev8))
    same_text = sum(text_hash(build_sentiment_text(dev11[k])) == text_hash(build_sentiment_text(dev8[k])) for k in shared)
    same_avail = sum(dev11[k].information_available_at == dev8[k].information_available_at for k in shared)
    share = lambda n, d: n / d if d else None  # noqa: E731
    return {"scope": "development articles only on both sides (information_available_at <= 2024-09-30T20:30:00Z)",
            "n_11b_dev_aapl": len(dev11), "n_phase8_dev_aapl": len(dev8), "n_shared": len(shared),
            "share_of_11b_in_phase8": share(len(shared), len(dev11)),
            "share_of_phase8_in_11b": share(len(shared), len(dev8)),
            "share_shared_identical_text_hash": share(same_text, len(shared)),
            "share_shared_identical_information_available_at": share(same_avail, len(shared)),
            "phase8_dataset": {"file": p8_path.name, "sha256": p8_sha},
            "pass_fail": False, "affects_results": False}


def develop(spec: Spec = FROZEN_SPEC, paths: Paths = DEFAULT_PATHS, workers: int = 12, log=print,
            data_comparison: Callable[[], dict] = amendment02_data_comparison) -> dict:
    reg = load_registry(spec, paths)
    for p, what in ((paths.results, "development results already exist; 11B.4 runs once"),
                    (paths.staging, "an incomplete development run exists; inspect it and move it aside manually"),
                    (paths.development, "a development folder exists without results; inspect it and move it aside")):
        if p.exists():
            raise Phase11BError(f"{p.name}: {what}")
    inputs = verified_inputs(spec)
    if inputs != reg["inputs"]:
        raise Phase11BError("inputs differ from the frozen 11B registry")
    if code_sha256() != reg["code_sha256"]:
        raise Phase11BError("research code differs from the code recorded at registry freeze")
    started = time.time()
    frames = load_frames(inputs, spec)
    log(f"[11B] frames: {len(frames)} stocks x {len(next(iter(frames.values())))} development sessions")
    data_comp = data_comparison()                       # independent of results: fail before hours of model fitting
    stock_res, panel_res = _run_jobs(frames, spec, workers, log)

    final_dir = paths.development
    paths.staging.mkdir(parents=True)
    final_art = {}

    def write_artifact(key: str, rel: Path, obj: dict) -> None:
        a = _write_new_json(paths.staging / rel, obj)
        final_art[key] = {"path": p9._rel(final_dir / rel), "sha256": a["sha256"]}

    for s in spec.universe:
        write_artifact(f"stock_{s}", Path("experiments") / f"stock_{s}.json",
                       {"symbol": s, "phase": PHASE, "matrix_sha256": reg["matrix_sha256"],
                        "frame_sha256": inputs["frames"][s]["sha256"], "runs": stock_res[s]})
    for (kind, fs, h), r in sorted(panel_res.items()):
        write_artifact(f"{kind}_{fs}_h{h}", Path("panel" if kind == "panel" else "sensitivity") / f"{kind}_{fs}_h{h}.json",
                       {"kind": kind, "feature_set": fs, "horizon": h, "phase": PHASE,
                        "matrix_sha256": reg["matrix_sha256"], **r})

    per_stock, panel, sens = _entries(stock_res, panel_res, spec, final_art)
    planned = reg["matrix"]["comparison_ids"]
    if sorted(e["id"] for e in per_stock) != sorted(planned["per_stock"]) or \
            sorted(e["id"] for e in panel) != sorted(planned["panel"]) or \
            sorted(e["id"] for e in sens) != sorted(planned["aapl_excluded_sensitivity"]):
        raise Phase11BError("results do not cover exactly the frozen 11B matrix")
    counts = (len(per_stock), len(panel), len(sens))
    if counts != tuple(spec.expected_counts):
        raise Phase11BError(f"comparison counts {counts} differ from {spec.expected_counts}")
    stage_holm_descriptive(per_stock, counts[0])
    stage_holm_descriptive(panel, counts[1])
    for e in sens:
        e["p_value_holm_11b_stage_descriptive"] = None
        e["final_qualification"] = "NEVER (descriptive sensitivity)"
    amendment02 = {"results_comparison": amendment02_results_comparison(per_stock) if "AAPL" in spec.universe else None,
                   "data_comparison": data_comp, "role": "descriptive only; never alters 11B results"}

    os.replace(paths.staging, final_dir)                                      # artifacts complete -> publish
    for k, a in final_art.items():
        if file_sha256(_resolve(a["path"])) != a["sha256"]:
            raise Phase11BError(f"artifact {k} changed while being published")
    summary = {fam: {"gate_passes": sum(e["gate_status"] == QUALIFIED for e in es),
                     "raw_p_below_alpha": sum(e["p_value_raw"] < ALPHA for e in es),
                     "stage_holm_p_below_alpha_descriptive": (
                         sum(e["p_value_holm_11b_stage_descriptive"] < HOLM_ALPHA for e in es) if fam != "sensitivity" else None)}
               for fam, es in (("per_stock", per_stock), ("panel", panel), ("sensitivity", sens))}
    results = {
        "document_type": "phase_11b_development_results", "phase": PHASE, "stage": STAGE, "status": "COMPLETE",
        "registry_sha256": file_sha256(paths.registry), "matrix_sha256": reg["matrix_sha256"],
        "inputs_sha256": reg["inputs_sha256"], "data_kind": reg["data_kind"],
        "runtime_seconds": round(time.time() - started, 1), "workers": workers, "git": git_info(),
        "environment": environment(),
        "counts": {"per_stock": counts[0], "panel": counts[1], "aapl_excluded_sensitivity": counts[2]},
        "summary": summary,
        "qualification": {"status": DEFERRED, "final_families": reg["matrix"]["families"]["final"],
                          "note": "no comparison is qualified in 11B.4; 11B.5 decides over 5,742 / 198"},
        "amendment_02": amendment02, "artifacts": final_art,
        "warnings_total": sum(r["warnings"]["total"] for runs in stock_res.values() for r in runs.values())
        + sum(r["warnings"]["total"] for r in panel_res.values()),
        "confirmation": "NOT_EVALUATED",
        "results": {"per_stock": per_stock, "panel": panel, "aapl_excluded_sensitivity": sens},
    }
    _write_new_json(paths.results, results)                                   # completion marker, last
    return results


# ============================================================
# Status / CLI
# ============================================================


def status_lines(spec: Spec = FROZEN_SPEC, paths: Paths = DEFAULT_PATHS) -> list[str]:
    lines = []
    try:
        inputs = verified_inputs(spec)
        lines.append("inputs           : verified")
    except Phase11BError as e:
        inputs = None
        lines.append(f"inputs           : NOT VERIFIED ({e})")
    for s in STEPS:
        ok, why = evidence_status(s, spec, inputs if s in REAL_STEPS else None, paths)
        lines.append(f"step {STEPS.index(s) + 1} {s:9s} : {'PASS' if ok else why}")
    if paths.registry.exists():
        try:
            reg = load_registry(spec, paths)
            lines.append(f"registry         : FROZEN {reg['matrix_sha256'][:16]}  counts {reg['matrix']['counts']}")
        except Phase11BError as e:
            lines.append(f"registry         : INVALID ({e})")
    else:
        lines.append("registry         : not frozen")
    lines.append(f"development      : {'COMPLETE' if paths.results.exists() else ('INCOMPLETE run present' if paths.staging.exists() else 'not run')}")
    lines.append("confirmation     : NOT_EVALUATED")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 11B Stage 11B.4 development research (B, C, D, F)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("freeze")
    v = sub.add_parser("validate")
    v.add_argument("--what", choices=list(STEPS), required=True)
    d = sub.add_parser("develop")
    d.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    if args.cmd == "status":
        print("\n".join(status_lines()))
    elif args.cmd == "validate":
        if args.what == "leakage":
            ev = validate_leakage()
        elif args.what == "smoke":
            ev = validate_smoke()
        elif args.what in ("aapl", "stock"):
            ev = validate_stock(args.what)
        else:
            ev = validate_panel()
        print(f"step {ev['order']} {ev['step']}: {ev['outcome']} ({ev['data_kind']})")
        print(json.dumps(ev["details"], indent=1, default=str))
    elif args.cmd == "freeze":
        reg = freeze()
        print(f"Frozen {DEFAULT_PATHS.registry.name}  matrix sha256 {reg['matrix_sha256']}  counts {reg['matrix']['counts']}")
    else:
        res = develop(workers=args.workers)
        print(json.dumps({k: res[k] for k in ("status", "counts", "summary", "runtime_seconds", "warnings_total")},
                         indent=1))
        print("Final qualification: DEFERRED to 11B.5 (families 5,742 / 198).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
