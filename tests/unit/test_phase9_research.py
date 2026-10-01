"""
Phase 9 research runner: pre-registration, matrix immutability, development /
holdout separation, qualifier freezing, registry completeness, gate preservation.
Uses tiny specs (fast linear / dummy candidates) - never the official matrix run.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import TimeSeriesSplit

import training.evaluation_harness as harness
import training.phase9_research as p9
from training.evaluation_harness import (
    CLASSIFICATION,
    REGRESSION,
    Candidate,
    HarnessConfig,
    build_targets,
    default_candidates,
    run_evaluation,
    walk_forward,
)

DATES = pd.bdate_range("2022-01-03", "2026-09-25")


def make_frame(signal: bool, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(DATES)
    x = rng.normal(0, 1, (n, 3))
    r = 0.01 * x[:, 0] + 0.001 * rng.normal(0, 1, n) if signal else 0.01 * rng.normal(0, 1, n)
    close = 100 * np.concatenate([[1.0], np.cumprod(1 + r[:-1])])    # Close[t+1] / Close[t] - 1 = r[t]
    return pd.DataFrame({"Date": DATES, "Close": close, "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2]})


def linear_candidates():
    d = {c.name: c for c in default_candidates()}
    return [d[n] for n in ("Mean Return", "Zero Return", "Linear Regression", "Always UP", "Base Rate",
                           "Logistic Regression")]


def clone_candidates():
    """Non-baseline candidates that ARE the baselines: can never beat them -> guaranteed no qualifier."""
    base = [c for c in default_candidates() if c.is_baseline]
    return base + [Candidate("Mean Clone", REGRESSION, lambda: DummyRegressor(strategy="mean")),
                   Candidate("Prior Clone", CLASSIFICATION, lambda: DummyClassifier(strategy="prior"))]


FS = {"A": {"families": ["test"], "columns": ["x1", "x2"]}, "B": {"families": ["test"], "columns": ["x1", "x2", "x3"]}}


def spec(factory=linear_candidates, horizons=(1,), n_splits=5, feature_sets=FS):
    return p9.ResearchSpec(feature_sets=feature_sets, horizons=horizons, candidates_factory=factory, n_splits=n_splits)


def frozen(tmp_path, s) -> Path:
    path = tmp_path / "registry.json"
    p9.freeze(path, inputs={"note": "test inputs"}, spec=s)
    return path


def develop(tmp_path, s, frame, **kw):
    path = frozen(tmp_path, s)
    return path, p9.run_development(path, frame, {"context": "test"}, spec=s, log=lambda *_: None, **kw)


# ---------- official matrix ----------

def test_official_matrix_is_the_approved_design():
    m = p9.OFFICIAL.matrix()
    counts = {k: v["n_features"] for k, v in m["feature_sets"].items()}
    assert counts == {"A": 20, "B": 35, "C": 43, "D": 58, "E": 29, "F": 65}
    assert m["horizons"] == [1, 3, 5]
    models = [c for c in m["candidates"] if not c["is_baseline"]]
    assert sorted(c["task"] for c in models).count(REGRESSION) == 6
    assert sorted(c["task"] for c in models).count(CLASSIFICATION) == 5
    assert len(m["experiments"]) == 6 * 3 * 11 == len({e["experiment_id"] for e in m["experiments"]})
    ev = m["evaluation"]
    assert (ev["gate_version"], ev["n_splits"], ev["gap"], ev["significance_level"]) == ("gate_v1", 20, "horizon", 0.05)
    assert ev["development_period"]["end"] == "2024-09-30"
    assert (ev["holdout_period"]["start"], ev["holdout_period"]["end"]) == ("2024-10-01", "2026-09-25")
    assert ev["multiple_comparisons"]["role"].startswith("informational")
    assert "sentiment_window_v1" in m["feature_definitions"] and "market_context_v1" in m["feature_definitions"]


def test_official_candidates_are_fixed_and_seeded():
    specs = {c["name"]: c for c in p9.OFFICIAL.matrix()["candidates"]}
    for name in ("RF Regressor", "HistGradientBoosting Regressor", "XGBoost Regressor", "LightGBM Regressor",
                 "Random Forest", "HistGradientBoosting", "XGBoost", "LightGBM"):
        assert specs[name]["random_seed"] == 42
    assert specs["HistGradientBoosting"]["params"]["early_stopping"] is False          # no random internal split
    assert specs["HistGradientBoosting Regressor"]["params"]["early_stopping"] is False
    assert not any("Search" in c["class"] for c in specs.values())                     # no tuning
    baselines = sorted(n for n, c in specs.items() if c["is_baseline"])
    assert baselines == ["Always UP", "Base Rate", "Mean Return", "Zero Return"]


def test_matrix_hash_is_stable():
    assert p9.matrix_sha256(p9.OFFICIAL.matrix()) == p9.matrix_sha256(p9.OFFICIAL.matrix())


# ---------- freezing / immutability ----------

def test_freeze_is_write_once(tmp_path):
    s = spec()
    path = frozen(tmp_path, s)
    reg = json.loads(path.read_text(encoding="utf-8"))
    assert reg["status"] == "FROZEN" and reg["development"] is None and reg["holdout"] is None
    assert set(reg["results"]) == {e["experiment_id"] for e in reg["matrix"]["experiments"]}
    with pytest.raises(p9.ResearchError, match="frozen once"):
        p9.freeze(path, inputs={}, spec=s)


def test_changed_matrix_is_refused(tmp_path):
    path = frozen(tmp_path, spec())
    with pytest.raises(p9.ResearchError, match="differs from the FROZEN"):
        p9.run_development(path, make_frame(True), {}, spec=spec(n_splits=6), log=lambda *_: None)
    with pytest.raises(p9.ResearchError, match="differs from the FROZEN"):
        p9.run_development(path, make_frame(True), {}, spec=spec(factory=clone_candidates), log=lambda *_: None)


def test_edited_registry_matrix_is_detected(tmp_path):
    path = frozen(tmp_path, spec())
    reg = json.loads(path.read_text(encoding="utf-8"))
    reg["matrix"]["horizons"] = [1, 2]
    path.write_text(json.dumps(reg), encoding="utf-8")
    with pytest.raises(p9.ResearchError, match="edited after freezing"):
        p9.load_registry(path)


# ---------- development ----------

def test_development_never_sees_holdout_rows(tmp_path):
    seen = []

    def spy_eval(**kw):
        seen.append(kw["data"]["Date"].max().date())
        return run_evaluation(**kw)

    develop(tmp_path, spec(horizons=(1, 3)), make_frame(True), run_eval=spy_eval)
    assert seen and all(d <= p9.DEV_END for d in seen)


def test_development_is_unchanged_by_holdout_data(tmp_path):
    f1 = make_frame(True)
    f2 = f1.copy()
    after = f2["Date"].dt.date > p9.DEV_END
    f2.loc[after, ["x1", "x2", "x3"]] = 0.0
    f2.loc[after, "Close"] = f2.loc[after, "Close"] * 3.0
    _, r1 = develop(tmp_path / "a", spec(), f1)
    _, r2 = develop(tmp_path / "b", spec(), f2)
    for exp_id, e in r1["results"].items():
        assert e["development"]["metrics"] == r2["results"][exp_id]["development"]["metrics"]
        assert e["development"]["gate"] == r2["results"][exp_id]["development"]["gate"]


def test_gate_result_is_the_harness_result(tmp_path):
    s = spec()
    frame = make_frame(True)
    _, reg = develop(tmp_path, s, frame)
    dev = p9.development_frame(frame, p9.DEV_END)
    direct = run_evaluation(data=dev[["Date", "Close", "x1", "x2"]], feature_columns=["x1", "x2"],
                            candidates=linear_candidates(), config=HarnessConfig(horizon=1, n_splits=5))
    report = json.loads(json.dumps(harness._json_safe(direct.report)))
    for m in report["models"]:
        if m["is_baseline"]:
            continue
        e = reg["results"][p9.experiment_id("A", 1, m["model"])]["development"]
        assert e["gate"] == m["qualification"] and e["metrics"] == m["metrics"]
    assert p9.qualify is harness.qualify and p9.run_evaluation is harness.run_evaluation


def test_registry_is_complete_and_artifacts_verified(tmp_path):
    path, reg = develop(tmp_path, spec(horizons=(1, 3)), make_frame(True))
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["status"] == "DEVELOPMENT_COMPLETE"
    assert on_disk["development"]["n_tests"] == len(on_disk["matrix"]["experiments"]) == 2 * 2 * 2
    for e in on_disk["results"].values():
        for k in ("experiment_id", "feature_set", "target", "horizon", "model", "task", "parameters", "random_seed"):
            assert k in e
        d = e["development"]
        for k in ("frame_period", "oos_period", "n_oos_rows", "n_splits", "gap", "metrics", "baseline_metrics",
                  "gate", "status", "p_value_raw", "p_value_holm", "artifact"):
            assert d[k] is not None or k == "random_seed", k
        assert d["gap"] == e["horizon"] and d["n_splits"] == 5
        assert d["p_value_holm"] >= d["p_value_raw"]
        art = Path(d["artifact"]["path"])
        assert p9.file_sha256(art) == d["artifact"]["sha256"]
    with pytest.raises(p9.ResearchError, match="already been run"):
        p9.run_development(path, make_frame(True), {}, spec=spec(horizons=(1, 3)), log=lambda *_: None)


def test_strong_signal_qualifies_and_is_frozen(tmp_path):
    _, reg = develop(tmp_path, spec(), make_frame(True))
    q = reg["development"]["qualifiers"]
    assert p9.experiment_id("A", 1, "Linear Regression") in q
    assert p9.experiment_id("A", 1, "Logistic Regression") in q
    assert reg["development"]["result"].endswith("DEVELOPMENT QUALIFIER(S)")


# ---------- holdout ----------

def test_holdout_requires_explicit_authorization(tmp_path):
    path, _ = develop(tmp_path, spec(), make_frame(True))
    with pytest.raises(p9.ResearchError, match="explicit authorization"):
        p9.run_holdout(path, make_frame(True), {}, confirm=False, spec=spec(), log=lambda *_: None)


def test_no_development_qualifiers_means_no_holdout(tmp_path):
    s = spec(factory=clone_candidates)
    path, reg = develop(tmp_path, s, make_frame(False))
    assert reg["development"]["qualifiers"] == []
    assert reg["development"]["result"] == p9.NO_DEVELOPMENT_QUALIFIERS
    reg = p9.run_holdout(path, None, None, confirm=True, spec=s, log=lambda *_: None)   # no data needed
    assert reg["holdout"]["status"] == "NOT_APPLICABLE"
    assert reg["holdout"]["result"] == p9.NO_DEVELOPMENT_QUALIFIERS
    assert not list((tmp_path / "experiments").glob("holdout_*"))
    with pytest.raises(p9.ResearchError, match="exactly once"):
        p9.run_holdout(path, None, None, confirm=True, spec=s, log=lambda *_: None)


def test_holdout_runs_only_frozen_qualifiers_once(tmp_path):
    s = spec()
    frame = make_frame(True)
    path, dev = develop(tmp_path, s, frame)
    qualifiers = dev["development"]["qualifiers"]
    reg = p9.run_holdout(path, frame, {"context": "test"}, confirm=True, spec=s, log=lambda *_: None)
    assert reg["holdout"]["evaluated"] == qualifiers
    files = sorted(p.name for p in (tmp_path / "experiments").glob("holdout_*"))
    assert files == sorted(f"holdout_{q}.json" for q in qualifiers)
    for exp_id, e in reg["results"].items():
        if exp_id not in qualifiers:
            assert e["holdout"] is None
            continue
        h = e["holdout"]
        lo, hi = date.fromisoformat(h["oos_period"]["start"]), date.fromisoformat(h["oos_period"]["end"])
        assert p9.HOLDOUT_START <= lo and hi <= p9.HOLDOUT_END
        assert e["status"] == ("CONFIRMED" if h["status"] == "QUALIFIED" else "NOT_CONFIRMED")
        art = json.loads(Path(h["artifact"]["path"]).read_text(encoding="utf-8"))
        for fold in art["folds"]:
            assert fold["train_end"] < fold["test_start"]
    assert p9.experiment_id("A", 1, "Linear Regression") in reg["holdout"]["confirmed"]
    with pytest.raises(p9.ResearchError, match="exactly once"):
        p9.run_holdout(path, frame, {}, confirm=True, spec=s, log=lambda *_: None)


@pytest.mark.parametrize("h", [1, 3, 5])
def test_holdout_splits_cover_the_holdout_with_gap(h):
    evaluable = build_targets(make_frame(True)[["Date", "Close"]], h)
    splits = p9.holdout_splits(evaluable["Date"], h, 20, p9.HOLDOUT_START, p9.HOLDOUT_END)
    d = evaluable["Date"].dt.date.to_numpy()
    expected = np.flatnonzero((d >= p9.HOLDOUT_START) & (d <= p9.HOLDOUT_END))
    assert np.array_equal(np.concatenate([t for _, t in splits]), expected)
    for train, test in splits:
        assert train[0] == 0 and train[-1] == test[0] - h - 1          # gap = horizon rows, like TimeSeriesSplit


def test_walk_forward_with_splits_reproduces_the_harness_loop():
    data = build_targets(make_frame(True)[["Date", "Close", "x1", "x2"]], 2)
    cands = linear_candidates()
    config = HarnessConfig(horizon=2, n_splits=6)
    expected = walk_forward(data, ["x1", "x2"], cands, config)
    splits = list(TimeSeriesSplit(n_splits=6, gap=2).split(data[["x1", "x2"]]))
    actual = p9.walk_forward_with_splits(data, ["x1", "x2"], cands, splits)
    pd.testing.assert_frame_equal(actual[0], expected[0])
    pd.testing.assert_frame_equal(actual[1], expected[1])
    assert actual[2] == expected[2]


# ---------- multiple comparisons ----------

def test_holm_adjustment():
    assert p9.holm_adjust([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert p9.holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert p9.holm_adjust([]) == []
    with pytest.raises(ValueError):
        p9.holm_adjust([1.2])


def test_existing_gate_and_baselines_unchanged():
    assert harness.GATE_VERSION == "gate_v1"
    assert HarnessConfig().significance_level == 0.05 and HarnessConfig(horizon=5).gap == 5
    names = [(c.name, c.task, c.is_baseline) for c in default_candidates()]
    assert names == [("Mean Return", REGRESSION, True), ("Zero Return", REGRESSION, True),
                     ("Linear Regression", REGRESSION, False), ("Ridge", REGRESSION, False),
                     ("Always UP", CLASSIFICATION, True), ("Base Rate", CLASSIFICATION, True),
                     ("Logistic Regression", CLASSIFICATION, False)]
