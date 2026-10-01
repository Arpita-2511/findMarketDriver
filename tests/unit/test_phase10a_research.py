"""
Phase 10A runner: matrix, F duplicate resolution, freeze immutability, input hashes,
harness equivalence, development cutoff / gap, gate preservation, holdout rules.
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
import training.phase10a_research as p10
import training.phase9_research as p9
from training.evaluation_harness import (
    CLASSIFICATION,
    REGRESSION,
    Candidate,
    HarnessConfig,
    default_candidates,
    run_evaluation,
)
from training.excess_targets import SPY_CLOSE, build_excess_targets

DATES = pd.bdate_range("2022-01-03", "2026-09-25")
PHASE9_MATRIX_SHA = "12e16f11ba886d1a61d7f91bf912478dc6850653373e3c1094afd26ae9b12545"
INPUTS = {"feature_store": {"path": "x", "sha256": "a" * 64}, "spy_snapshot": {"path": "s", "sha256": "b" * 64}}


def make_frame(signal: bool, seed: int = 0, spy_constant: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(DATES)
    x = rng.normal(0, 1, (n, 3))
    s = np.zeros(n) if spy_constant else 0.01 * rng.normal(0, 1, n)
    e = 0.01 * x[:, 0] + 0.001 * rng.normal(0, 1, n) if signal else 0.01 * rng.normal(0, 1, n)
    a = s + e                                                     # excess r[t] = a[t] - s[t] = e[t]
    aapl = 100 * np.concatenate([[1.0], np.cumprod(1 + a[:-1])])
    spy = 400 * np.concatenate([[1.0], np.cumprod(1 + s[:-1])])
    return pd.DataFrame({"Date": DATES, "Close": aapl, SPY_CLOSE: spy, "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2]})


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
    return p10.Spec10A(feature_sets=FS, horizons=horizons, candidates_factory=factory, n_splits=n_splits)


def frozen(tmp_path, s):
    path = tmp_path / "registry.json"
    p10.freeze(path, inputs=INPUTS, validation={"note": "test"}, spec=s)
    return path


def develop(tmp_path, s, frame, **kw):
    path = frozen(tmp_path, s)
    return path, p10.run_development(path, frame, INPUTS, spec=s, log=lambda *_: None, **kw)


# ---------- official matrix ----------

def test_official_matrix_198():
    m = p10.OFFICIAL.matrix()
    assert {k: v["n_features"] for k, v in m["feature_sets"].items()} == \
        {"A": 20, "B": 35, "C": 43, "D": 58, "E": 29, "F": 64}
    assert m["horizons"] == [1, 3, 5] and m["phase"] == "10A"
    ids = [e["experiment_id"] for e in m["experiments"]]
    assert len(ids) == 198 == len(set(ids)) and all(i.startswith("p10a_") for i in ids)
    tasks = [e["task"] for e in m["experiments"]]
    assert tasks.count(REGRESSION) == 108 and tasks.count(CLASSIFICATION) == 90
    assert "p10a_A_h1_rf_regressor" in ids
    ev = m["evaluation"]
    assert (ev["gate_version"], ev["n_splits"], ev["gap"], ev["significance_level"]) == ("gate_v1", 20, "horizon", 0.05)
    assert ev["development_period"]["start"] == "2017-02-01" and ev["development_period"]["end"] == "2024-09-30"
    assert (ev["confirmation_holdout"]["start"], ev["confirmation_holdout"]["end"]) == ("2024-10-01", "2026-09-25")
    assert m["targets"]["3"]["regression"]["name"] == "future_excess_return_3d"


def test_f_duplicate_resolution():
    f = p10.FEATURE_SETS["F"]["columns"]
    assert "sw_count_1" not in f and "event__article_count" in f
    assert len(f) == len(set(f)) == 64
    assert set(f) == (set(p9.FEATURE_SETS["F"]["columns"]) - {"sw_count_1"})          # otherwise Phase 9's F
    assert not any(c.startswith("sentiment__") for c in f)                             # no cumulative sentiment in F
    r = p10.OFFICIAL.matrix()["f_duplicate_resolution"]
    assert (r["dropped"], r["retained"]) == ("sw_count_1", "event__article_count")


def test_phase9_matrix_is_untouched():
    assert p9.matrix_sha256(p9.OFFICIAL.matrix()) == PHASE9_MATRIX_SHA


def test_baselines_keep_harness_names_and_gate_functions():
    names = sorted(c["name"] for c in p10.OFFICIAL.matrix()["candidates"] if c["is_baseline"])
    assert names == ["Always UP", "Base Rate", "Mean Return", "Zero Return"]
    assert p10.qualify is harness.qualify and p10.walk_forward is harness.walk_forward
    assert p10.pooled_metrics is harness.pooled_metrics and harness.GATE_VERSION == "gate_v1"
    assert p10.BASELINE_LABELS["Base Rate"].startswith("Excess Direction Base Rate")


def test_matrix_hash_deterministic():
    assert p10.matrix_sha256(p10.OFFICIAL.matrix()) == p10.matrix_sha256(p10.OFFICIAL.matrix())


# ---------- freeze / immutability / inputs ----------

def test_freeze_write_once_and_pending(tmp_path):
    path = frozen(tmp_path, spec())
    reg = json.loads(path.read_text(encoding="utf-8"))
    assert reg["status"] == "FROZEN" and reg["phase"] == "10A" and reg["development"] is None
    assert reg["determinism"] == p10.DETERMINISM_NOT_VERIFIED
    assert all(r["status"] == "PENDING" for r in reg["results"].values())
    with pytest.raises(p10.Phase10AError, match="frozen once"):
        p10.freeze(path, inputs=INPUTS, validation={}, spec=spec())


def test_changed_matrix_or_edited_registry_refused(tmp_path):
    path = frozen(tmp_path, spec())
    with pytest.raises(p10.Phase10AError, match="differs from the FROZEN"):
        p10.run_development(path, make_frame(True), INPUTS, spec=spec(n_splits=6), log=lambda *_: None)
    reg = json.loads(path.read_text(encoding="utf-8"))
    reg["matrix"]["horizons"] = [1, 2]
    path.write_text(json.dumps(reg), encoding="utf-8")
    with pytest.raises(p10.Phase10AError, match="edited after freezing"):
        p10.load_registry(path)


def test_input_hash_mismatch_stops(tmp_path):
    path = frozen(tmp_path, spec())
    other = json.loads(json.dumps(INPUTS))
    other["spy_snapshot"]["sha256"] = "c" * 64
    with pytest.raises(p10.Phase10AError, match="spy_snapshot"):
        p10.run_development(path, make_frame(True), other, spec=spec(), log=lambda *_: None)
    with pytest.raises(p10.Phase10AError, match="input hashes differ"):
        p10.run_target_diagnostics(path, make_frame(True), other, spec=spec())


# ---------- freeze validation ----------

def _validation_frame():
    f = make_frame(True)
    f["event__article_count"] = np.arange(len(f)) % 7
    f["sw_count_1"] = f["event__article_count"].astype("float64")
    return f


def _bars(f):
    return pd.DataFrame({"Date": f["Date"], "Close": f["Close"]})


def test_freeze_validation_checks_duplicate_and_close():
    f = _validation_frame()
    s = p10.Spec10A(feature_sets={"F": {"families": ["t"], "columns": ["x1", "event__article_count"]}},
                    horizons=(1,), candidates_factory=linear_candidates, n_splits=5)
    prov = {"frame": {"rows_before_full_sentiment_window": 0}}
    v = p10.freeze_validation(f, prov, _bars(f), s, rebuilt_frame=None)
    assert v["duplicate_check"]["identical"] and v["duplicate_check"]["dropped"] == "sw_count_1"
    assert all(v["construction_determinism"]["development_targets_identical"].values())
    assert v["target_availability"]["1"]["development_labelled_rows"] == v["development"]["rows"] - 1
    g = f.copy()
    g.loc[10, "sw_count_1"] += 1
    with pytest.raises(p10.Phase10AError, match="not identical"):
        p10.freeze_validation(g, prov, _bars(g), s)
    bars = _bars(f)
    bars.loc[5, "Close"] *= 1.01
    with pytest.raises(p10.Phase10AError, match="snapshot Close"):
        p10.freeze_validation(f, prov, bars, s)


# ---------- evaluation equivalence / cutoff / gap ----------

def test_constant_spy_reproduces_run_evaluation_exactly():
    f = make_frame(True, spy_constant=True)
    cands, cols = linear_candidates(), ["x1", "x2"]
    mine = p10.evaluate_excess(f, cols, cands, horizon=3, n_splits=5)
    ref = run_evaluation(data=f[["Date", "Close", *cols]], feature_columns=cols, candidates=linear_candidates(),
                         config=HarnessConfig(horizon=3, n_splits=5))
    ref_models = {m["model"]: m for m in ref.report["models"]}
    for m in mine["models"]:
        assert m["metrics"] == ref_models[m["model"]]["metrics"]
        assert m["qualification"] == ref_models[m["model"]]["qualification"]
    assert mine["folds"] == ref.report["folds"]
    assert mine["evaluation_period"] == ref.report["evaluation_period"]


@pytest.mark.parametrize("h", [1, 3, 5])
def test_folds_respect_gap(h):
    f = make_frame(True)
    out = p10.evaluate_excess(f, ["x1"], linear_candidates(), horizon=h, n_splits=5)
    dates = list(build_excess_targets(f[["Date", "Close", SPY_CLOSE]], h)["Date"].dt.date.astype(str))
    for fold in out["folds"]:
        assert dates.index(fold["test_start"]) - dates.index(fold["train_end"]) == h + 1


def test_development_cut_before_targets(tmp_path):
    seen = []

    def spy_eval(dev, cols, cands, **kw):
        seen.append(dev["Date"].max().date())
        return p10.evaluate_excess(dev, cols, cands, **kw)

    develop(tmp_path, spec(horizons=(1, 5)), make_frame(True), evaluate=spy_eval)
    assert seen and all(d <= p10.DEV_END for d in seen)


def test_development_independent_of_post_cutoff_prices(tmp_path):
    f1 = make_frame(True)
    f2 = f1.copy()
    after = f2["Date"].dt.date > p10.DEV_END
    f2.loc[after, "Close"] *= 2.0
    f2.loc[after, SPY_CLOSE] *= 0.5
    f2.loc[after, ["x1", "x2", "x3"]] = 0.0
    _, r1 = develop(tmp_path / "a", spec(), f1)
    _, r2 = develop(tmp_path / "b", spec(), f2)
    for k, e in r1["results"].items():
        assert e["development"]["metrics"] == r2["results"][k]["development"]["metrics"]
        assert e["development"]["gate"] == r2["results"][k]["development"]["gate"]


def test_registry_records_everything(tmp_path):
    path, reg = develop(tmp_path, spec(horizons=(1, 3)), make_frame(True))
    disk = json.loads(path.read_text(encoding="utf-8"))
    assert disk["development"]["n_tests"] == 2 * 2 * 2
    for e in disk["results"].values():
        d = e["development"]
        assert d["gap"] == e["horizon"] and d["n_splits"] == 5 and d["p_value_holm"] >= d["p_value_raw"]
        assert e["target"]["name"].startswith(("future_excess_return_", "excess_direction_"))
        art = json.loads(Path(d["artifact"]["path"]).read_text(encoding="utf-8"))
        assert art["inputs"] == INPUTS and "warnings" in art and art["matrix_sha256"] == disk["matrix_sha256"]
        assert p10.file_sha256(Path(d["artifact"]["path"])) == d["artifact"]["sha256"]
    with pytest.raises(p10.Phase10AError, match="already been run"):
        p10.run_development(path, make_frame(True), INPUTS, spec=spec(horizons=(1, 3)), log=lambda *_: None)


# ---------- qualification / holdout ----------

def test_zero_qualifiers_close_the_holdout_without_reading_it(tmp_path):
    s = spec(factory=clone_candidates)
    path, reg = develop(tmp_path, s, make_frame(False))
    assert reg["development"]["qualifiers"] == [] and reg["development"]["result"] == p10.NO_DEVELOPMENT_QUALIFIERS
    assert reg["status"] == "COMPLETE" and reg["holdout"]["status"] == "NOT_APPLICABLE"
    with pytest.raises(p10.Phase10AError):
        p10.run_holdout(path, make_frame(False), INPUTS, confirm=True, spec=s, log=lambda *_: None)
    assert not list((tmp_path / "experiments").glob("holdout_*"))


def test_holdout_requires_authorization_and_runs_once(tmp_path):
    s = spec()
    f = make_frame(True)
    path, dev = develop(tmp_path, s, f)
    q = dev["development"]["qualifiers"]
    assert p10.experiment_id("A", 1, "Linear Regression") in q and dev["status"] == "DEVELOPMENT_COMPLETE"
    with pytest.raises(p10.Phase10AError, match="explicit authorization"):
        p10.run_holdout(path, f, INPUTS, confirm=False, spec=s, log=lambda *_: None)
    reg = p10.run_holdout(path, f, INPUTS, confirm=True, spec=s, log=lambda *_: None)
    assert reg["holdout"]["evaluated"] == q
    for exp_id in q:
        h = reg["results"][exp_id]["holdout"]
        lo, hi = date.fromisoformat(h["oos_period"]["start"]), date.fromisoformat(h["oos_period"]["end"])
        assert p10.HOLDOUT_START <= lo and hi <= p10.HOLDOUT_END
    assert p10.experiment_id("A", 1, "Linear Regression") in reg["holdout"]["confirmed"]
    with pytest.raises(p10.Phase10AError, match="exactly once"):
        p10.run_holdout(path, f, INPUTS, confirm=True, spec=s, log=lambda *_: None)


def test_target_diagnostics_development_only(tmp_path):
    s = spec(horizons=(1, 3))
    path = frozen(tmp_path, s)
    f = make_frame(True)
    diag = p10.run_target_diagnostics(path, f, INPUTS, spec=s)
    for h, d in diag["horizons"].items():
        assert d["all_spot_checks_match"]
        assert date.fromisoformat(d["last_label_close_date"]) <= p10.DEV_END
        assert set(d["holdout_structural"]) == {"holdout_rows", "labelled_rows", "first_date", "last_labelled_date"}
    with pytest.raises(p10.Phase10AError, match="already recorded"):
        p10.run_target_diagnostics(path, f, INPUTS, spec=s)
