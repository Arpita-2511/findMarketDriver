"""
Phase 12 Stage 1 signal diagnostics - synthetic tests.

Frames are fabricated (frozen column layout, random values, development-period dates). No real
frame, article, model or sealed-period row is used.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import training.phase11b_research as r
import training.phase12_signal_diagnostics as d
from training.phase11b_feature_build import FRAME_COLUMNS
from training.phase9_research import canonical_json, holm_adjust


def make_frame(n=600, seed=0, start="2018-01-02", news_rate=1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({"Date": pd.bdate_range(start, periods=n)})
    f["Close"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    f["SPY_Close"] = 300 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    for c in FRAME_COLUMNS[3:]:
        f[c] = rng.normal(0, 1, n)
    f["event__article_count"] = rng.poisson(news_rate, n).astype(float)
    f["sw_count_1"] = f["event__article_count"]
    f["Volatility"] = np.abs(rng.normal(0.01, 0.003, n))
    return f[list(FRAME_COLUMNS)]


def frames(k=3, n=600, **kw) -> dict:
    return {s: make_frame(n, seed=i, **kw) for i, s in enumerate(("AAA", "BBB", "CCC")[:k])}


def _features():
    fams = d.feature_families()
    feats = [c for v in fams.values() for c in v]
    return fams, feats, ["control" if c in fams[d.CONTROL] else "candidate" for c in feats]


def _run(fr, shift=None):
    fams, feats, kinds = _features()
    samples, dates, _ = d.build_samples(fr, feats)
    return d.run_tests(samples, feats, kinds, len(dates), shift=shift)


# ============================================================
# Families and targets
# ============================================================


def test_feature_families_have_the_frozen_sizes_and_controls_are_separate():
    fams, feats, kinds = _features()
    assert {k: len(v) for k, v in fams.items()} == {"sentiment_window": 13, "technical": 20, "market_context": 9,
                                                    "event": 23, d.CONTROL: 15}
    assert all(c.startswith("sentiment__") for c in fams[d.CONTROL])
    assert kinds.count("control") == 15 and kinds.count("candidate") == 65
    assert len(set(feats)) == 80


@pytest.mark.parametrize("h", d.HORIZONS)
def test_targets_are_the_frozen_excess_return_aligned_forward(h):
    f = make_frame(50)
    t = d.build_targets(f, h)
    assert len(t) == len(f) - h and list(t["Date"]) == list(f["Date"][:len(f) - h])
    c, s = f["Close"].to_numpy(), f["SPY_Close"].to_numpy()
    want = (c[h:] / c[:-h] - 1) - (s[h:] / s[:-h] - 1)
    np.testing.assert_allclose(t["t_return"], want, rtol=0, atol=1e-15)
    np.testing.assert_allclose(t["t_magnitude"], np.abs(want), rtol=0, atol=1e-15)
    np.testing.assert_array_equal(t["t_direction"], (want > 0).astype(float))


def test_zero_excess_return_is_down():
    f = make_frame(20)
    f["SPY_Close"] = f["Close"]                                           # identical returns -> excess exactly 0
    t = d.build_targets(f, 1)
    assert (t["t_return"] == 0).all() and (t["t_direction"] == 0).all()


def test_rows_after_the_development_end_are_refused():
    fr = frames(1)
    fr["AAA"].loc[len(fr["AAA"]) - 1, "Date"] = pd.Timestamp("2024-10-01")
    with pytest.raises(d.Phase12Error, match="development end"):
        d.check_development_only(fr)


def test_feature_at_t_is_paired_with_the_target_from_t_onward():
    """A feature equal to the future target correlates perfectly; the realized (past) return does not."""
    fr = frames(1, n=700)
    f = fr["AAA"]
    c, s = f["Close"], f["SPY_Close"]
    f["RSI"] = ((c.shift(-1) / c - 1) - (s.shift(-1) / s - 1)).fillna(0.0)    # leaks the h=1 target
    f["Return_1"] = ((c / c.shift(1) - 1) - (s / s.shift(1) - 1)).fillna(0.0)  # information available at t
    res = _run(fr)
    row = lambda feat: res[(res.level == "per_stock") & (res.scope == "all_days") & (res.target == "return")
                           & (res.horizon == 1) & (res.feature == feat)].iloc[0]
    assert row("RSI")["rho"] == pytest.approx(1.0)
    assert abs(row("Return_1")["rho"]) < 0.15


# ============================================================
# Statistics
# ============================================================


def test_bartlett_sum_known_value():
    assert d.bartlett_sum(np.array([1.0, 2.0, 3.0]), 1)[0] == pytest.approx(14 + 2 * 0.5 * 8)
    assert d.bartlett_sum(np.array([1.0, 2.0, 3.0]), 0)[0] == pytest.approx(14)


def test_hac_spearman_rho_equals_scipy_and_lag0_is_the_iid_sandwich():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(400, 3))
    y = 0.3 * x[:, 0] + rng.normal(size=400)
    res = d.hac_spearman(x, y, 0)
    for j in range(3):
        assert res["rho"][j] == pytest.approx(stats.spearmanr(x[:, j], y).statistic, abs=1e-12)
    ux, uy = d.rank_standardize(x), d.rank_standardize(y)
    e = uy[:, None] - ux * res["rho"]
    np.testing.assert_allclose(res["se"], np.sqrt(((ux * e) ** 2).sum(axis=0)) / 400, rtol=1e-12)
    assert res["t"][0] > 5 and abs(res["t"][1]) < 4


def test_ties_use_average_ranks_and_constant_columns_are_nan():
    z = d.rank_standardize(np.array([[1.0, 5.0], [1.0, 5.0], [2.0, 5.0]]))
    assert z[0, 0] == z[1, 0] and np.isnan(z[:, 1]).all()
    res = d.hac_spearman(np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]]), np.array([1, 3, 2, 4.0]), 1)
    assert np.isnan(res["rho"][1]) and np.isnan(res["t"][1])


def test_driscoll_kraay_identical_stocks_add_no_information():
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(300, 2)), rng.normal(size=300)
    ux, uy, pos = d.rank_standardize(x), d.rank_standardize(y), np.arange(300)
    one = d._slope_hac([(ux, uy, pos)], 300, 5)
    two = d._slope_hac([(ux, uy, pos), (ux, uy, pos)], 300, 5)
    np.testing.assert_allclose(two["rho"], one["rho"])
    np.testing.assert_allclose(two["se"], one["se"])                       # perfectly dependent stocks: same SE
    assert (two["n"] == 600).all()


def test_driscoll_kraay_independent_stocks_shrink_the_se():
    rng = np.random.default_rng(6)
    units = [(d.rank_standardize(rng.normal(size=(500, 1))), d.rank_standardize(rng.normal(size=500)),
              np.arange(500)) for _ in range(4)]
    single = d._slope_hac(units[:1], 500, 5)["se"][0]
    pooled = d._slope_hac(units, 500, 5)["se"][0]
    assert pooled < single * 0.7                                           # about 1 / sqrt(4)


def test_lag_rule():
    assert d.lag_for(1927, 1) == 13 and d.lag_for(1923, 5) == 13 and d.lag_for(300, 5) == 10


# ============================================================
# Samples, families, Holm, placebo
# ============================================================


def test_news_day_scope_and_minimum_sample_eligibility():
    fr = frames(2, n=600)
    fr["BBB"]["event__article_count"] = 0.0
    fr["BBB"].loc[:99, "event__article_count"] = 1.0                       # 100 news days < MIN_N
    fr["BBB"]["sw_count_1"] = fr["BBB"]["event__article_count"]
    _, feats, _ = _features()
    samples, _, sizes = d.build_samples(fr, feats)
    lab = d.build_targets(fr["AAA"], 1)
    assert sizes["AAA|h1|news_days"] == int((lab["event__article_count"] > 0).sum())
    assert sizes["BBB|h1|news_days"] == 100
    res = _run(fr)
    bbb = res[(res.symbol == "BBB") & (res.scope == "news_days")]
    assert not bbb["eligible"].any()
    assert res[(res.symbol == "AAA") & (res.scope == "news_days")]["eligible"].any()


def test_constant_feature_is_ineligible():
    fr = frames(1)
    fr["AAA"]["event__legal_event_count"] = 0.0
    res = _run(fr)
    assert not res[res.feature == "event__legal_event_count"]["eligible"].any()


def test_partial_volatility_family_excludes_volatility_itself():
    res = _run(frames(2))
    pv = res[res.kind.str.endswith("_partial_vol")]
    assert len(pv) and "Volatility" not in set(pv.feature) and set(pv.target) == {"magnitude"}
    assert set(pv.level) == {"pooled"}


def test_holm_is_applied_within_each_family_only_to_eligible_rows():
    df = pd.DataFrame({"family": ["a", "a", "a", "a", "b", "b"], "symbol": [None] * 6, "horizon": [1, 1, 5, 5, 1, 5],
                       "feature": ["f1", "f2", "f1", "f2", "f1", "f1"], "p": [0.01, 0.04, 0.03, 0.005, 0.02, 0.5],
                       "eligible": [True, True, True, True, True, False]})
    out = d.apply_holm(df)
    np.testing.assert_allclose(out.loc[:3, "p_holm"], [0.03, 0.06, 0.06, 0.02])
    assert out.loc[4, "p_holm"] == pytest.approx(0.02) and np.isnan(out.loc[5, "p_holm"])
    assert holm_adjust([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_families_separate_levels_scopes_targets_and_controls():
    res = _run(frames(2))
    fams = set(res["family"])
    assert "pooled|all_days|return|candidate" in fams and "pooled|all_days|return|control" in fams
    assert "per_stock|news_days|magnitude|candidate" in fams
    assert "pooled|all_days|magnitude|candidate_partial_vol" in fams
    ctrl = res[res.kind == "control"]
    assert set(ctrl.feature) == set(d.feature_families()[d.CONTROL])


def test_placebo_shift_breaks_a_planted_signal_and_is_calibrated_on_null_data():
    fr = frames(3, n=800)
    for f in fr.values():
        c, s = f["Close"], f["SPY_Close"]
        fut = ((c.shift(-1) / c - 1) - (s.shift(-1) / s - 1)).fillna(0.0)
        f["RSI"] = fut + np.random.default_rng(1).normal(0, 0.01, len(f))   # planted h=1 signal
    real = _run(fr)
    sel = lambda df: df[(df.level == "pooled") & (df.scope == "all_days") & (df.target == "return")
                        & (df.horizon == 1) & (df.feature == "RSI")].iloc[0]
    assert sel(real)["t"] > 10
    for frac in (0.1, 0.5, 0.9):
        assert abs(sel(_run(fr, shift=frac))["t"]) < 4.5
    null = _run(frames(3, n=800), shift=0.3)
    el = null[null.eligible & (null.level == "per_stock")]
    assert 0.0 <= (el["p"] < 0.05).mean() < 0.12                         # nominal 5%


def test_placebo_uses_the_same_offset_for_every_stock():
    fr = frames(2)
    fams, feats, kinds = _features()
    samples, dates, _ = d.build_samples(fr, feats)
    s0 = [s for s in samples if s.h == 1 and s.scope == "all_days"]
    shifts = {int(round(0.3 * len(s.y_full["return"]))) for s in s0}
    assert len(shifts) == 1


def test_calibration_columns_follow_their_definitions(monkeypatch):
    monkeypatch.setattr(d, "PLACEBO_FRACTIONS", (0.2, 0.4, 0.6, 0.8))
    df, summ = d.diagnose(frames(2), log=lambda m: None)
    el = df[df.eligible]
    np.testing.assert_allclose(el["t_calibrated"], el["t"] / np.maximum(1.0, el["placebo_rms_t"]))
    np.testing.assert_allclose(el["p_calibrated"], 2 * stats.norm.sf(el["t_calibrated"].abs()))
    assert el["p_empirical"].between(1 / 5, 1).all()
    assert df.loc[~df.eligible, "p_holm_calibrated"].isna().all()
    assert (el["p_holm_calibrated"] >= el["p_calibrated"] - 1e-15).all()
    assert {x["n_shifts"] for x in summ["placebo"]} == {4}


def _persistent(n, rng, phi=0.998):
    e = rng.normal(size=n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + e[i]
    return out


def test_calibration_tames_spurious_correlation_between_persistent_series(monkeypatch):
    """Independent but very persistent feature and volatility: raw HAC over-rejects, calibration does not."""
    monkeypatch.setattr(d, "PLACEBO_FRACTIONS", tuple(np.linspace(0.1, 0.9, 30)))
    rng = np.random.default_rng(11)
    fr = {}
    for s in ("AAA", "BBB", "CCC"):
        f = make_frame(900, seed=int(rng.integers(1000)))
        vol = 0.01 * np.exp(0.6 * _persistent(900, rng) / 10)          # slow volatility regimes
        f["Close"] = 100 * np.exp(np.cumsum(vol * rng.normal(size=900)))
        f["SPY_Close"] = 300 * np.exp(np.cumsum(0.002 * rng.normal(size=900)))
        for c in d.feature_families()["technical"]:
            if c != "Volatility":
                f[c] = _persistent(900, rng)                             # independent of the volatility path
        fr[s] = f
    df, _ = d.diagnose(fr, log=lambda m: None)
    sel = df[df.eligible & (df.level == "pooled") & (df.scope == "all_days") & (df.target == "magnitude")
             & (df.kind == "candidate") & df.feature.isin(d.feature_families()["technical"])
             & (df.feature != "Volatility")]
    raw, cal = (sel["p"] < 0.05).mean(), (sel["p_calibrated"] < 0.05).mean()
    assert raw > 0.2 and cal < raw / 2


# ============================================================
# Reproducibility and write-once outputs
# ============================================================


def test_diagnose_is_reproducible(monkeypatch):
    monkeypatch.setattr(d, "PLACEBO_FRACTIONS", (0.2, 0.6))
    fr = frames(3)
    a_df, a_sum = d.diagnose({k: v.copy() for k, v in fr.items()}, log=lambda m: None)
    b_df, b_sum = d.diagnose({k: v.copy() for k, v in fr.items()}, log=lambda m: None)
    pd.testing.assert_frame_equal(a_df, b_df)
    assert canonical_json(json.loads(json.dumps(a_sum, default=str))) == \
        canonical_json(json.loads(json.dumps(b_sum, default=str)))


def test_run_writes_once_and_status_verifies(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "PLACEBO_FRACTIONS", (0.25, 0.75))
    fr = frames(3)
    monkeypatch.setattr(r, "verified_inputs", lambda spec: {"feature_run_id": "synthetic", "frames": {}})
    monkeypatch.setattr(r, "load_frames", lambda inputs, spec: {k: v.copy() for k, v in fr.items()})
    out = tmp_path / "phase12" / "stage1"
    doc = d.run(out, log=lambda m: None)
    assert sorted(p.name for p in out.iterdir()) == ["manifest.json", "summary.json", "tests.csv"]
    assert doc["sealed_data_read"] is False and doc["status"].startswith("EXPLORATORY")
    assert "verified" in d.status_lines(out)[0]
    with pytest.raises(d.Phase12Error, match="write-once"):
        d.run(out, log=lambda m: None)
    first = (out / "tests.csv").read_bytes()
    out2 = tmp_path / "again"
    d.run(out2, log=lambda m: None)
    assert (out2 / "tests.csv").read_bytes() == first                     # byte-identical rerun
