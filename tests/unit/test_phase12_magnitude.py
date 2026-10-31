"""
Phase 12 Stage 2 magnitude prediction - synthetic tests.

Frames are fabricated (frozen column layout, development-period dates). A small Config keeps the
walk-forward fast; the frozen Config is only checked for its pre-registered values.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import training.phase12_magnitude as m
from tests.unit.test_phase12_signal_diagnostics import make_frame

CFG = m.Config(min_train=150, test_len=40, min_last_fold=15, rf_trees=20, lgb_trees=30, min_leaf_pooled=40,
               min_leaf_stock=20, n_repro_stocks=2)


def planted(n=520, k=3, strength=0.9, seed=0) -> dict:
    """|excess return| at t+1 grows with Volume_Change_1 at t; everything else is noise."""
    out = {}
    for i in range(k):
        rng = np.random.default_rng(seed + i)
        f = make_frame(n, seed=seed + i)
        vc = rng.normal(size=n)
        sigma = 0.01 * np.exp(strength * vc)                                   # monotone in Volume_Change_1
        r = np.concatenate([[0.0], sigma[:-1] * rng.normal(size=n - 1)])         # return t-1 -> t uses VC at t-1
        f["Volume_Change_1"] = vc
        f["Close"] = 100 * np.exp(np.cumsum(r))
        f["SPY_Close"] = 300 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
        out[("AAA", "BBB", "CCC", "DDD")[i]] = f
    return out


# ============================================================
# Protocol
# ============================================================


def test_frozen_protocol_values():
    p = m.protocol()
    c = p["config"]
    assert (c["min_train"], c["test_len"], c["min_rel_mse_improvement"], c["min_blocks_improved"],
            c["min_share_stocks_improved"], c["alpha"]) == (504, 63, 0.01, 3, 0.6, 0.05)
    assert p["features"] == list(m.FEATURES) and len(m.FEATURES) == 8
    assert set(m.CANDIDATES) == {"ridge", "random_forest", "lightgbm"}
    assert "direction" in p["excluded"]
    assert m.protocol_sha256() == m.protocol_sha256()


def test_protocol_is_written_once_and_run_requires_it(tmp_path):
    with pytest.raises(m.Phase12S2Error, match="protocol.json not found"):
        m.run(tmp_path, CFG, frames=planted(), log=lambda s: None)
    m.write_protocol(tmp_path, CFG)
    with pytest.raises(m.Phase12S2Error, match="written once"):
        m.write_protocol(tmp_path, CFG)


def test_run_refuses_a_changed_protocol(tmp_path):
    m.write_protocol(tmp_path, CFG)
    p = tmp_path / "protocol.json"
    doc = json.loads(p.read_text())
    doc["config"]["min_rel_mse_improvement"] = 0.0
    p.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    with pytest.raises(m.Phase12S2Error, match="cannot change"):
        m.run(tmp_path, CFG, frames=planted(), log=lambda s: None)
    with pytest.raises(m.Phase12S2Error, match="cannot change"):
        m.run(tmp_path, m.Config(min_train=151), frames=planted(), log=lambda s: None)


# ============================================================
# Data, folds, leakage
# ============================================================


@pytest.mark.parametrize("h", (1, 5))
def test_folds_respect_minimum_training_window_and_horizon_gap(h):
    folds = m.make_folds(400, h, CFG)
    assert folds[0] == (150, 150 + h, 190 + h)
    for (tr, st, en), nxt in zip(folds, folds[1:] + [None]):
        assert (tr - 1) + h < st and tr == st - h
        if nxt:
            assert nxt[1] == en
    assert folds[-1][2] == 400 and folds[-1][2] - folds[-1][1] >= CFG.min_last_fold


def test_short_final_fold_is_merged():
    folds = m.make_folds(150 + 1 + 40 + 5, 1, CFG)                         # 5 leftover sessions
    assert len(folds) == 1 and folds[0][2] - folds[0][1] == 45


def test_panel_targets_are_absolute_frozen_excess_returns():
    fr = planted(n=60, k=2)
    panel = m.build_panel(fr, 5)
    f = fr["AAA"]
    c, s = f["Close"].to_numpy(), f["SPY_Close"].to_numpy()
    want = np.abs((c[5:] / c[:-5] - 1) - (s[5:] / s[:-5] - 1))
    np.testing.assert_allclose(panel[panel.symbol == "AAA"]["y"], want, atol=1e-15)
    assert "symbol" not in m.FEATURES and set(m.model_columns("ridge")) == set(m.FEATURES)


def test_sealed_period_rows_are_refused():
    fr = planted(n=60, k=1)
    fr["AAA"].loc[59, "Date"] = pd.Timestamp("2024-10-01")
    with pytest.raises(Exception, match="development end"):
        m.build_panel(fr, 1)


def test_non_finite_inputs_stop_the_run():
    fr = planted(n=60, k=1)
    fr["AAA"].loc[3, "Volume_Change_5"] = np.nan
    with pytest.raises(m.Phase12S2Error, match="non-finite"):
        m.build_panel(fr, 1)


def test_test_period_targets_never_influence_predictions():
    panel = m.build_panel(planted(), 1)
    folds = m.make_folds(int(panel.pos.max()) + 1, 1, CFG)
    a = m.walk_forward(panel, 1, CFG, "pooled")
    st = folds[0][1]
    changed = panel.copy()
    changed.loc[changed.pos >= st, "y"] = 99.0                             # fold-0 test targets and later
    b = m.walk_forward(changed, 1, CFG, "pooled")
    f0 = a.fold == 0
    for name in m.MODEL_ROLES:
        np.testing.assert_array_equal(a.loc[f0, name].to_numpy(), b.loc[f0, name].to_numpy())


def test_pooled_scaling_uses_training_rows_only():
    panel = m.build_panel(planted(), 1)
    train, test = panel[panel.pos < 200], panel[panel.pos >= 200].copy()
    _, _, st1 = m.pooled_scale(train, test)
    test[list(m.STOCK_SCALED)] = 1e6
    _, te2, st2 = m.pooled_scale(train, test)
    for s in st1:
        pd.testing.assert_series_equal(st1[s][0], st2[s][0])
    assert set(st1) == {"AAA", "BBB", "CCC"}
    assert "Volatility" not in m.STOCK_SCALED


def test_reference_baseline_uses_only_volatility_features():
    panel = m.build_panel(planted(), 1)
    a = m.walk_forward(panel, 1, CFG, "pooled", models=(m.REFERENCE, "ridge"))
    other = panel.copy()
    for c in m.STOCK_SCALED:
        other[c] = other[c] * -3 + 7
    b = m.walk_forward(other, 1, CFG, "pooled", models=(m.REFERENCE, "ridge"))
    np.testing.assert_array_equal(a[m.REFERENCE], b[m.REFERENCE])
    assert not np.array_equal(a["ridge"], b["ridge"])


def test_predictions_are_floored_at_zero_and_rows_identical_across_models():
    preds = m.walk_forward(m.build_panel(planted(), 5), 5, CFG, "pooled")
    for name in m.MODEL_ROLES:
        assert (preds[name] >= 0).all()
    assert not preds.duplicated(["symbol", "pos"]).any()


def test_quantile_clipper_learns_bounds_from_training_data():
    c = m.QuantileClipper((0.1, 0.9)).fit(np.arange(11.0)[:, None])
    assert c.transform(np.array([[-5.0], [50.0]])).ravel().tolist() == [1.0, 9.0]


# ============================================================
# Statistics and qualification
# ============================================================


def test_dm_one_sided():
    rng = np.random.default_rng(1)
    assert m.dm_one_sided(0.5 + rng.normal(size=500), 5)["p"] < 1e-6
    assert m.dm_one_sided(-0.5 + rng.normal(size=500), 5)["p"] > 0.999
    assert m.dm_one_sided(np.zeros(50), 3) == {"mean_diff": 0.0, "t": 0.0, "p": 1.0}
    assert m.lag_for(1422, 1) == 12 and m.lag_for(1414, 5) == 12 and m.lag_for(27, 5) == 5


def _cmp(**kw):
    c = {"rel_mse_improvement": 0.01, "rel_mae_improvement": 0.0, "dm": {"p_2L": 0.01}, "blocks_improved": 3,
         "share_stocks_improved": 0.6, "rel_mse_without_best_stock": 0.001, "rel_mse_without_best_fold": 0.001}
    c.update(kw)
    return c


def test_qualification_thresholds_and_boundaries():
    assert m.qualify(_cmp(), 0.049, True, True, m.FROZEN_CONFIG, pooled=True)["qualified"]
    for kw, crit in ((dict(rel_mse_improvement=0.0099), "Q1_improvement"),
                     (dict(rel_mae_improvement=-1e-9), "Q1_improvement"),
                     (dict(dm={"p_2L": 0.05}), "Q2_statistics"),
                     (dict(blocks_improved=2), "Q3_periods"),
                     (dict(share_stocks_improved=0.59), "Q5_breadth"),
                     (dict(rel_mse_without_best_stock=0.0), "Q5_breadth"),
                     (dict(rel_mse_without_best_fold=-0.1), "Q5_breadth")):
        q = m.qualify(_cmp(**kw), 0.01, True, True, m.FROZEN_CONFIG, pooled=True)
        assert not q["qualified"] and q["failed"] == [crit]
    assert m.qualify(_cmp(), 0.05, True, True, m.FROZEN_CONFIG, pooled=True)["failed"] == ["Q2_statistics"]
    assert m.qualify(_cmp(), 0.01, False, True, m.FROZEN_CONFIG, pooled=True)["failed"] == ["Q4_leakage"]
    assert m.qualify(_cmp(), 0.01, True, False, m.FROZEN_CONFIG, pooled=True)["failed"] == ["Q6_reproducibility"]
    per = m.qualify(_cmp(share_stocks_improved=0.0), 0.01, True, True, m.FROZEN_CONFIG, pooled=False)
    assert per["Q5_breadth"] is None and per["qualified"]


def test_calibration_table():
    rng = np.random.default_rng(2)
    y = np.abs(rng.normal(size=1000))
    preds = pd.DataFrame({"y": y, "a": y * 0.5 + 0.1})
    cal = m.calibration(preds, "a")
    means = [d["mean_pred"] for d in cal["deciles"]]
    assert means == sorted(means) and sum(d["n"] for d in cal["deciles"]) == 1000
    assert cal["top5pct_actual"]["ratio_pred_to_actual"] < 1                  # shrunk predictions understate large moves
    assert cal["spearman_pred_actual"] == pytest.approx(1.0)


# ============================================================
# End to end
# ============================================================


def test_experiment_finds_a_planted_signal_and_families_are_fixed():
    exp = m.experiment(planted(), CFG, log=lambda s: None)
    assert exp["family_sizes"] == {"primary": 6, "secondary": 3 * 3 * 2}
    ridge = next(c for c in exp["primary_comparisons"] if c["model"] == "ridge" and c["horizon"] == 1)
    assert ridge["rel_mse_improvement"] > 0.05 and ridge["dm"]["p_holm"] < 0.05
    for h in (1, 5):
        assert exp["horizons"][h]["reproducibility"]["pooled_identical"]
        assert exp["horizons"][h]["reproducibility"]["per_stock_identical"]
    q = next(q for q in exp["qualification"] if q["scope"] == "pooled" and q["model"] == "ridge" and q["horizon"] == 1)
    assert set(q) >= {"Q1_improvement", "Q2_statistics", "Q3_periods", "Q4_leakage", "Q5_breadth", "Q6_reproducibility"}


def test_no_signal_does_not_qualify():
    fr = planted(strength=0.0)
    exp = m.experiment(fr, CFG, log=lambda s: None)
    assert not any(q["qualified"] for q in exp["qualification"] if q["scope"] == "pooled")


def test_run_writes_once_traces_inputs_and_is_reproducible(tmp_path):
    fr = planted()
    for d in (tmp_path / "a", tmp_path / "b"):
        m.write_protocol(d, CFG)
        doc = m.run(d, CFG, frames={k: v.copy() for k, v in fr.items()}, inputs_sha="synthetic", log=lambda s: None)
        assert doc["protocol_sha256"] == m.protocol_sha256(CFG) and doc["inputs_sha256"] == "synthetic"
        assert doc["sealed_data_read"] is False
        assert "verified" in m.status_lines(d)[1]
    with pytest.raises(m.Phase12S2Error, match="write-once"):
        m.run(tmp_path / "a", CFG, frames=fr, log=lambda s: None)
    for f in ("per_stock_metrics.csv", "per_stock_comparisons.csv"):
        assert (tmp_path / "a" / "results" / f).read_bytes() == (tmp_path / "b" / "results" / f).read_bytes()
    res = json.loads((tmp_path / "a" / "results" / "results.json").read_text())
    qualified = [q for q in res["qualification"] if q["scope"] == "pooled" and q["qualified"]]
    assert len(res["finalists"]) == len(qualified)
    files = json.loads((tmp_path / "a" / "results" / "manifest.json").read_text())["files"]
    assert all(f"finalists/pooled_{q['model']}_h{q['horizon']}.joblib" in files for q in qualified)
