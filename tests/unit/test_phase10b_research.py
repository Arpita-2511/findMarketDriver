"""
Phase 10B runner: matrix, reuse of the frozen Phase 10A design, freeze immutability, inputs,
equivalence with Phase 10A, cutoff/gap, gate preservation, holdout rules.
Tiny specs with fast candidates only - never the official 198-experiment run.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

import training.evaluation_harness as harness
import training.phase10a_research as p10a
import training.phase10b_research as p10b
import training.phase9_research as p9
from training.evaluation_harness import CLASSIFICATION, REGRESSION, Candidate, default_candidates
from training.excess_targets import SPY_CLOSE
from training.normalized_targets import VOL_COLUMN, build_normalized_targets

DATES = pd.bdate_range("2022-01-03", "2026-09-25")
PHASE9_SHA = "12e16f11ba886d1a61d7f91bf912478dc6850653373e3c1094afd26ae9b12545"
PHASE10A_SHA = "fbd763e7980232bb1bf7409616f8dbbb0af9e8b0c70857a680cb89aad1da6a45"
INPUTS = {"feature_store": {"path": "x", "sha256": "a" * 64}, "spy_snapshot": {"path": "s", "sha256": "b" * 64}}


def make_frame(signal: bool, seed: int = 0, vol_one: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(DATES)
    x = rng.normal(0, 1, (n, 3))
    s = 0.01 * rng.normal(0, 1, n)
    e = 0.01 * x[:, 0] + 0.001 * rng.normal(0, 1, n) if signal else 0.01 * rng.normal(0, 1, n)
    aapl = 100 * np.concatenate([[1.0], np.cumprod(1 + (s + e)[:-1])])
    spy = 400 * np.concatenate([[1.0], np.cumprod(1 + s[:-1])])
    vol = np.ones(n) if vol_one else 0.005 + 0.01 * rng.random(n)
    return pd.DataFrame({"Date": DATES, "Close": aapl, SPY_CLOSE: spy, VOL_COLUMN: vol,
                         "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2]})


def linear_candidates():
    d = {c.name: c for c in default_candidates()}
    return [d[n] for n in ("Mean Return", "Zero Return", "Linear Regression", "Always UP", "Base Rate",
                           "Logistic Regression")]


def clone_candidates():
    base = [c for c in default_candidates() if c.is_baseline]
    return base + [Candidate("Mean Clone", REGRESSION, lambda: DummyRegressor(strategy="mean")),
                   Candidate("Prior Clone", CLASSIFICATION, lambda: DummyClassifier(strategy="prior"))]


FS = {"A": {"families": ["t"], "columns": ["x1", "x2"]}, "B": {"families": ["t"], "columns": ["x1", "x2", "x3"]}}


def spec(factory=linear_candidates, horizons=(1,), n_splits=5):
    return p10b.Spec10B(feature_sets=FS, horizons=horizons, candidates_factory=factory, n_splits=n_splits)


def frozen(tmp_path, s):
    path = tmp_path / "registry.json"
    p10b.freeze(path, inputs=INPUTS, validation={"note": "test"}, spec=s)
    return path


def develop(tmp_path, s, frame, **kw):
    path = frozen(tmp_path, s)
    return path, p10b.run_development(path, frame, INPUTS, spec=s, log=lambda *_: None, **kw)


# ---------- matrix ----------

def test_official_matrix_198_reuses_phase10a_design():
    m = p10b.OFFICIAL.matrix()
    assert m["phase"] == "10B" and m["horizons"] == [1, 3, 5]
    assert {k: v["n_features"] for k, v in m["feature_sets"].items()} == \
        {"A": 20, "B": 35, "C": 43, "D": 58, "E": 29, "F": 64}
    assert m["feature_sets"] == p10a.OFFICIAL.matrix()["feature_sets"]
    assert m["f_duplicate_resolution"]["dropped"] == "sw_count_1"
    assert [c["params"] for c in m["candidates"]] == [c["params"] for c in p10a.OFFICIAL.matrix()["candidates"]]
    ids = [e["experiment_id"] for e in m["experiments"]]
    assert len(ids) == 198 == len(set(ids)) and all(i.startswith("p10b_") for i in ids)
    tasks = [e["task"] for e in m["experiments"]]
    assert tasks.count(REGRESSION) == 108 and tasks.count(CLASSIFICATION) == 90
    ev = m["evaluation"]
    assert (ev["gate_version"], ev["n_splits"], ev["gap"], ev["significance_level"]) == ("gate_v1", 20, "horizon", 0.05)
    assert ev["development_period"]["end"] == "2024-09-30"
    assert ev["confirmation_holdout"]["name"] == "Phase-10B confirmation holdout"
    assert m["volatility_definition"]["window_sessions"] == 20
    assert m["targets"]["5"]["regression"]["name"] == "normalized_excess_return_5d"


def test_previous_matrices_untouched_and_hash_stable():
    assert p9.matrix_sha256(p9.OFFICIAL.matrix()) == PHASE9_SHA
    assert p10a.matrix_sha256(p10a.OFFICIAL.matrix()) == PHASE10A_SHA
    assert p10b.matrix_sha256(p10b.OFFICIAL.matrix()) == p10b.matrix_sha256(p10b.OFFICIAL.matrix())
    assert p10b.matrix_sha256(p10b.OFFICIAL.matrix()) != PHASE10A_SHA


def test_gate_components_are_the_harness():
    assert p10b.qualify is harness.qualify and p10b.walk_forward is harness.walk_forward
    assert p10b.pooled_metrics is harness.pooled_metrics and harness.GATE_VERSION == "gate_v1"
    names = sorted(c["name"] for c in p10b.OFFICIAL.matrix()["candidates"] if c["is_baseline"])
    assert names == ["Always UP", "Base Rate", "Mean Return", "Zero Return"]


# ---------- freeze / inputs ----------

def test_freeze_write_once_changed_matrix_and_inputs(tmp_path):
    path = frozen(tmp_path, spec())
    reg = json.loads(path.read_text(encoding="utf-8"))
    assert reg["status"] == "FROZEN" and reg["phase"] == "10B"
    with pytest.raises(p10b.Phase10BError, match="frozen once"):
        p10b.freeze(path, inputs=INPUTS, validation={}, spec=spec())
    with pytest.raises(p10b.Phase10BError, match="differs from the FROZEN"):
        p10b.run_development(path, make_frame(True), INPUTS, spec=spec(n_splits=6), log=lambda *_: None)
    other = json.loads(json.dumps(INPUTS))
    other["feature_store"]["sha256"] = "f" * 64
    with pytest.raises(p10b.Phase10BError, match="input hashes differ"):
        p10b.run_development(path, make_frame(True), other, spec=spec(), log=lambda *_: None)
    with pytest.raises(p10a.Phase10AError, match="not a Phase 10A registry"):
        p10a.load_registry(path)                                     # a 10B registry is not a 10A registry


# ---------- equivalence ----------

def test_volatility_one_reproduces_phase10a_exactly():
    f = make_frame(True, vol_one=True)
    cands, cols = linear_candidates(), ["x1", "x2"]
    mine = p10b.evaluate_normalized(f, cols, cands, horizon=3, n_splits=5)
    ref = p10a.evaluate_excess(f, cols, linear_candidates(), horizon=3, n_splits=5)
    for a, b in zip(mine["models"], ref["models"]):
        assert (a["metrics"], a["qualification"]) == (b["metrics"], b["qualification"])
    assert mine["folds"] == ref["folds"]


def test_classification_identical_to_phase10a_for_any_positive_volatility():
    f = make_frame(True)
    cols = ["x1", "x2"]
    mine = {m["model"]: m for m in p10b.evaluate_normalized(f, cols, linear_candidates(), horizon=1, n_splits=5)["models"]}
    ref = {m["model"]: m for m in p10a.evaluate_excess(f, cols, linear_candidates(), horizon=1, n_splits=5)["models"]}
    for name in ("Always UP", "Base Rate", "Logistic Regression"):
        assert mine[name]["metrics"] == ref[name]["metrics"]
        assert mine[name]["qualification"] == ref[name]["qualification"]


@pytest.mark.parametrize("h", [1, 3, 5])
def test_folds_respect_gap(h):
    f = make_frame(True)
    out = p10b.evaluate_normalized(f, ["x1"], linear_candidates(), horizon=h, n_splits=5)
    dates = list(build_normalized_targets(f[["Date", "Close", SPY_CLOSE, VOL_COLUMN]], h)["Date"].dt.date.astype(str))
    for fold in out["folds"]:
        assert dates.index(fold["test_start"]) - dates.index(fold["train_end"]) == h + 1


# ---------- development ----------

def test_development_cutoff_and_independence(tmp_path):
    seen = []

    def spy_eval(dev, cols, cands, **kw):
        seen.append(dev["Date"].max().date())
        return p10b.evaluate_normalized(dev, cols, cands, **kw)

    f1 = make_frame(True)
    _, r1 = develop(tmp_path / "a", spec(horizons=(1, 5)), f1, evaluate=spy_eval)
    assert seen and all(d <= p10b.DEV_END for d in seen)
    f2 = f1.copy()
    after = f2["Date"].dt.date > p10b.DEV_END
    f2.loc[after, ["Close", SPY_CLOSE, VOL_COLUMN]] *= 2.0
    f2.loc[after, ["x1", "x2", "x3"]] = 0.0
    _, r2 = develop(tmp_path / "b", spec(horizons=(1, 5)), f2)
    for k, e in r1["results"].items():
        assert e["development"]["metrics"] == r2["results"][k]["development"]["metrics"]


def test_registry_and_artifacts_auditable(tmp_path):
    path, reg = develop(tmp_path, spec(horizons=(1, 3)), make_frame(True))
    disk = json.loads(path.read_text(encoding="utf-8"))
    assert disk["development"]["n_tests"] == 8
    for e in disk["results"].values():
        d = e["development"]
        assert d["gap"] == e["horizon"] and d["p_value_holm"] >= d["p_value_raw"]
        assert e["target"]["name"].startswith(("normalized_excess_return_", "normalized_excess_direction_"))
        art = json.loads(Path(d["artifact"]["path"]).read_text(encoding="utf-8"))
        for key in ("inputs", "matrix_sha256", "features", "folds", "fold_metrics", "warnings", "baseline_metrics",
                    "training_period", "evaluation_period", "target"):
            assert key in art
        assert p10b.file_sha256(Path(d["artifact"]["path"])) == d["artifact"]["sha256"]


def test_zero_qualifiers_close_holdout_without_reading(tmp_path):
    s = spec(factory=clone_candidates)
    path, reg = develop(tmp_path, s, make_frame(False))
    assert reg["development"]["result"] == p10b.NO_DEVELOPMENT_QUALIFIERS
    assert reg["status"] == "COMPLETE" and reg["holdout"]["status"] == "NOT_APPLICABLE"
    with pytest.raises(p10b.Phase10BError):
        p10b.run_holdout(path, make_frame(False), INPUTS, confirm=True, spec=s, log=lambda *_: None)
    assert not list((tmp_path / "experiments").glob("holdout_*"))


def test_holdout_requires_authorization_and_runs_once(tmp_path):
    s = spec()
    f = make_frame(True)
    path, dev = develop(tmp_path, s, f)
    assert p10b.experiment_id("A", 1, "Linear Regression") in dev["development"]["qualifiers"]
    with pytest.raises(p10b.Phase10BError, match="explicit authorization"):
        p10b.run_holdout(path, f, INPUTS, confirm=False, spec=s, log=lambda *_: None)
    reg = p10b.run_holdout(path, f, INPUTS, confirm=True, spec=s, log=lambda *_: None)
    for exp_id in reg["holdout"]["evaluated"]:
        h = reg["results"][exp_id]["holdout"]
        assert p10b.HOLDOUT_START <= date.fromisoformat(h["oos_period"]["start"])
        assert date.fromisoformat(h["oos_period"]["end"]) <= p10b.HOLDOUT_END
    with pytest.raises(p10b.Phase10BError, match="exactly once"):
        p10b.run_holdout(path, f, INPUTS, confirm=True, spec=s, log=lambda *_: None)


def test_target_diagnostics_development_only(tmp_path):
    s = spec(horizons=(1, 3))
    path = frozen(tmp_path, s)
    diag = p10b.run_target_diagnostics(path, make_frame(True), INPUTS, spec=s)
    for d in diag["horizons"].values():
        assert d["all_spot_checks_match"] and d["direction_equals_excess_direction"]
        assert date.fromisoformat(d["last_label_close_date"]) <= p10b.DEV_END
        assert set(d["holdout_structural"]) == {"holdout_rows", "labelled_rows", "first_date", "last_labelled_date"}
    with pytest.raises(p10b.Phase10BError, match="already recorded"):
        p10b.run_target_diagnostics(path, make_frame(True), INPUTS, spec=s)
