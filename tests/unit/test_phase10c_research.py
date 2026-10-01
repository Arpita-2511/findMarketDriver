"""Phase 10C runner: matrix, gate preservation, regime-conditioned evaluation, Holm family, holdout rules."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

import training.evaluation_harness as harness
import training.phase10a_research as p10a
import training.phase10b_research as p10b
import training.phase10c_research as p10c
import training.phase9_research as p9
from training import regimes as rg
from training.evaluation_harness import (
    CLASSIFICATION,
    REGRESSION,
    Candidate,
    HarnessConfig,
    default_candidates,
    pooled_metrics,
    qualify,
)
from training.excess_targets import SPY_CLOSE

DATES = pd.bdate_range("2022-01-03", "2026-09-25")
INPUTS = {"feature_store": {"path": "x", "sha256": "a" * 64}}


def make_frame(signal: bool, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(DATES)
    x = rng.normal(0, 1, (n, 3))
    s = 0.01 * rng.normal(0, 1, n)
    e = 0.01 * x[:, 0] + 0.001 * rng.normal(0, 1, n) if signal else 0.01 * rng.normal(0, 1, n)
    return pd.DataFrame({
        "Date": DATES, "Close": 100 * np.concatenate([[1.0], np.cumprod(1 + (s + e)[:-1])]),
        SPY_CLOSE: 400 * np.concatenate([[1.0], np.cumprod(1 + s[:-1])]),
        rg.AAPL_VOL: 0.01 + 0.01 * rng.random(n), rg.SPY_VOL: 0.005 + 0.01 * rng.random(n),
        rg.SPY_TREND: 1.0 + 0.05 * rng.normal(0, 1, n), rg.SPY_RET20: 0.03 * rng.normal(0, 1, n),
        "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2]})


def linear_candidates():
    d = {c.name: c for c in default_candidates()}
    return [d[n] for n in ("Mean Return", "Zero Return", "Linear Regression", "Always UP", "Base Rate",
                           "Logistic Regression")]


def clone_candidates():
    base = [c for c in default_candidates() if c.is_baseline]
    return base + [Candidate("Mean Clone", REGRESSION, lambda: DummyRegressor(strategy="mean")),
                   Candidate("Prior Clone", CLASSIFICATION, lambda: DummyClassifier(strategy="prior"))]


FS = {"A": {"families": ["t"], "columns": ["x1", "x2"]}}


def spec(factory=linear_candidates, horizons=(1,), n_splits=5, min_rows=100):
    return p10c.Spec10C(feature_sets=FS, horizons=horizons, candidates_factory=factory, n_splits=n_splits,
                        min_state_oos=min_rows)


def frozen(tmp_path, s):
    path = tmp_path / "registry.json"
    p10c.freeze(path, inputs=INPUTS, validation={"note": "test"}, spec=s)
    return path


def develop(tmp_path, s, frame, **kw):
    path = frozen(tmp_path, s)
    return path, p10c.run_development(path, frame, INPUTS, spec=s, log=lambda *_: None, **kw)


def test_official_matrix():
    m = p10c.OFFICIAL.matrix()
    assert m["n_base_experiments"] == 198 and m["n_comparisons_max"] == 198 * 12 == 2376
    assert len(set(m["comparisons"])) == 2376
    assert {k: v["n_features"] for k, v in m["feature_sets"].items()} == \
        {"A": 20, "B": 35, "C": 43, "D": 58, "E": 29, "F": 64}
    assert m["feature_sets"] == p10a.OFFICIAL.matrix()["feature_sets"]
    ev = m["evaluation"]
    assert ev["gate_version"] == "gate_v1" and ev["n_splits"] == 20 and ev["min_state_oos_rows"] == 100
    assert "Holm-adjusted p < 0.05" in ev["qualification_criterion"]
    assert m["regimes"]["state_count"] == 12 and "R5_spy_vol_x_trend" in m["regimes"]["families"]
    assert m["targets"]["1"]["regression"]["name"] == "future_excess_return_1d"


def test_earlier_matrices_untouched():
    assert p9.matrix_sha256(p9.OFFICIAL.matrix()) == "12e16f11ba886d1a61d7f91bf912478dc6850653373e3c1094afd26ae9b12545"
    assert p10a.matrix_sha256(p10a.OFFICIAL.matrix()) == "fbd763e7980232bb1bf7409616f8dbbb0af9e8b0c70857a680cb89aad1da6a45"
    assert p10b.matrix_sha256(p10b.OFFICIAL.matrix()) == "101dd0fd888b28b101a510adecf4d4ef11ea25500e3540cefa457900548bffd3"


def test_gate_components_unchanged():
    assert p10c.qualify is harness.qualify and p10c.walk_forward is harness.walk_forward
    assert p10c.pooled_metrics is harness.pooled_metrics and p10c.make_splitter is harness.make_splitter


def test_unconditional_reference_and_folds_equal_phase10a():
    f = make_frame(True)
    out = p10c.evaluate_regimes(f, ["x1", "x2"], linear_candidates(), horizon=3, n_splits=5)
    ref = p10a.evaluate_excess(f, ["x1", "x2"], linear_candidates(), horizon=3, n_splits=5)
    assert out["folds"] == ref["folds"]                                          # leakage item 10
    for a, b in zip(out["unconditional_reference"]["models"], ref["models"]):
        assert (a["metrics"], a["qualification"]) == (b["metrics"], b["qualification"])


def test_state_result_is_qualify_on_exactly_the_state_rows():
    f = make_frame(True)
    cands = linear_candidates()
    out = p10c.evaluate_regimes(f, ["x1"], cands, horizon=1, n_splits=5, min_rows=10)
    from training.excess_targets import build_excess_targets
    ev = build_excess_targets(f[["Date", "Close", SPY_CLOSE, *rg.REGIME_INPUT_COLUMNS, "x1"]], 1)
    cfg = HarnessConfig(horizon=1, n_splits=5)
    preds, _, _ = harness.walk_forward(ev, ["x1"], cands, cfg)
    splits = list(harness.make_splitter(cfg).split(ev[["x1"]]))
    states, _ = rg.assign_oos_states({c: ev[c].to_numpy() for c in rg.REGIME_INPUT_COLUMNS}, splits)
    rows = {r for r, s in states["R3_spy_trend"].items() if s == "NEGATIVE_TREND"}
    sub = preds[preds["row"].isin(rows)]
    gate = qualify(sub, cands, pooled_metrics(sub, cands), cfg)
    got = {m["model"]: m["qualification"] for m in out["regimes"]["R3_spy_trend"]["NEGATIVE_TREND"]["models"]}
    assert got == gate and out["regimes"]["R3_spy_trend"]["NEGATIVE_TREND"]["n_oos"] == len(rows)
    n_states = sum(out["regimes"]["R5_spy_vol_x_trend"][s]["n_oos"] for s in rg.FAMILIES["R5_spy_vol_x_trend"]["states"])
    assert n_states == out["evaluation_period"]["n_oos_rows"]


def test_regime_labels_independent_of_targets():
    f1 = make_frame(True)
    f2 = f1.copy()
    f2["Close"] = f2["Close"] * np.linspace(1.0, 3.0, len(f2))          # different targets, same regime inputs
    a = p10c.evaluate_regimes(f1, ["x1"], linear_candidates(), horizon=1, n_splits=5)
    b = p10c.evaluate_regimes(f2, ["x1"], linear_candidates(), horizon=1, n_splits=5)
    assert a["regime_folds"] == b["regime_folds"]                                 # leakage item 9


def test_insufficient_state_is_not_tested(tmp_path):
    f = make_frame(True)
    f.loc[:, rg.SPY_TREND] = 1.1                                           # every row POSITIVE_TREND
    path, reg = develop(tmp_path, spec(), f)
    neg = [c for c in reg["results"].values() if c["state"] == "NEGATIVE_TREND"]
    assert neg and all(c["status"] == p10c.INSUFFICIENT_DATA and not c["eligible"] for c in neg)
    assert reg["development"]["family_size"] == sum(c["eligible"] for c in reg["results"].values())


def test_strong_signal_qualifies_with_holm_and_registry_complete(tmp_path):
    path, reg = develop(tmp_path, spec(), make_frame(True))
    d = reg["development"]
    assert d["n_comparisons"] == 2 * 12 and set(reg["results"]) == set(reg["matrix"]["comparisons"])
    q = d["qualifiers"]
    assert any(c.startswith("p10c_A_h1_linear_regression__") for c in q)
    for cid in q:
        c = reg["results"][cid]
        assert c["gate_status"] == "QUALIFIED" and c["p_value_holm"] < 0.05 and c["n_oos"] >= 100
    eligible = [c for c in reg["results"].values() if c["eligible"]]
    assert [c["p_value_holm"] for c in eligible] == pytest.approx(p10c.holm_adjust([c["p_value_raw"] for c in eligible]))
    art = json.loads(Path(eligible[0]["artifact"]).read_text(encoding="utf-8"))
    for key in ("inputs", "matrix_sha256", "regime_folds", "regimes", "unconditional_reference", "folds", "warnings"):
        assert key in art
    assert reg["status"] == "DEVELOPMENT_COMPLETE"


def test_zero_qualifiers_close_holdout_without_reading(tmp_path):
    s = spec(factory=clone_candidates)
    path, reg = develop(tmp_path, s, make_frame(False))
    assert reg["development"]["result"] == p10c.NO_DEVELOPMENT_QUALIFIERS
    assert reg["status"] == "COMPLETE" and reg["holdout"]["status"] == "NOT_APPLICABLE"
    with pytest.raises(p10c.Phase10CError):
        p10c.run_holdout(path, make_frame(False), INPUTS, confirm=True, spec=s, log=lambda *_: None)
    assert not list((tmp_path / "experiments").glob("holdout_*"))


def test_holdout_authorization_and_once(tmp_path):
    s = spec()
    f = make_frame(True)
    path, _ = develop(tmp_path, s, f)
    with pytest.raises(p10c.Phase10CError, match="explicit authorization"):
        p10c.run_holdout(path, f, INPUTS, confirm=False, spec=s, log=lambda *_: None)
    reg = p10c.run_holdout(path, f, INPUTS, confirm=True, spec=s, log=lambda *_: None)
    assert reg["holdout"]["status"] == "COMPLETE"
    with pytest.raises(p10c.Phase10CError, match="exactly once"):
        p10c.run_holdout(path, f, INPUTS, confirm=True, spec=s, log=lambda *_: None)


def test_freeze_immutability_and_inputs(tmp_path):
    path = frozen(tmp_path, spec())
    with pytest.raises(p10c.Phase10CError, match="frozen once"):
        p10c.freeze(path, inputs=INPUTS, validation={}, spec=spec())
    with pytest.raises(p10c.Phase10CError, match="differs from the FROZEN"):
        p10c.run_development(path, make_frame(True), INPUTS, spec=spec(min_rows=50), log=lambda *_: None)
    with pytest.raises(p10c.Phase10CError, match="input hashes differ"):
        p10c.run_development(path, make_frame(True), {"other": {}}, spec=spec(), log=lambda *_: None)


def test_regime_diagnostics_feature_only(tmp_path):
    s = spec(horizons=(1, 3))
    path = frozen(tmp_path, s)
    diag = p10c.run_regime_diagnostics(path, make_frame(True), INPUTS, spec=s)
    for d in diag["horizons"].values():
        for fam, counts in d["pooled_oos_state_counts"].items():
            assert sum(counts.values()) == d["oos_rows"]
    assert diag["expected_family_size"] == sum(2 * d["n_eligible_states"] for d in diag["horizons"].values())
    with pytest.raises(p10c.Phase10CError, match="already recorded"):
        p10c.run_regime_diagnostics(path, make_frame(True), INPUTS, spec=s)
