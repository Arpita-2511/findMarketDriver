"""Phase 11A: cross-stock price-only research (offline; synthetic data except the input-hash checks)."""

import json

import numpy as np
import pandas as pd
import pytest

import training.phase10a_research as p10a
import training.phase11_design as p11d
import training.phase11a_research as a
from training.evaluation_harness import QUALIFIED, diebold_mariano
from training.excess_targets import SPY_CLOSE

FAST = [c for c in a.candidates() if c.name in ("Mean Return", "Zero Return", "Ridge", "Always UP", "Base Rate",
                                                 "Logistic Regression")]
FAST_PANEL = FAST + [a.PER_STOCK_MEAN, a.PER_STOCK_RATE]
COLS = ["f1", "f2", "f3"]


def frame(seed: int, n: int = 160, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n)
    close = 100 * np.cumprod(1 + drift + rng.normal(0, 0.01, n))
    spy = 300 * np.cumprod(1 + rng.normal(0, 0.008, n))
    f = pd.DataFrame({"Date": dates, "Close": close, SPY_CLOSE: spy})
    for c in COLS:
        f[c] = rng.normal(size=n)
    return f


def frames(k: int = 3) -> dict:
    return {s: frame(i, drift=0.002 * i) for i, s in enumerate(["AAA", "BBB", "CCC"][:k])}


def bars(seed: int, end: str, start: str = "2016-06-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, "2017-06-30")
    rng = np.random.default_rng(seed)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates)))
    o = c * (1 + rng.normal(0, 0.002, len(dates)))
    b = pd.DataFrame({"Date": dates.astype("datetime64[ns]"), "Open": o, "High": np.maximum(o, c) * 1.01,
                      "Low": np.minimum(o, c) * 0.99, "Close": c, "Volume": rng.integers(1e6, 2e6, len(dates))})
    return b[b["Date"] <= pd.Timestamp(end)].reset_index(drop=True)


# ---------------- universe, matrix, families ----------------


def test_universe_and_feature_sets():
    assert len(a.UNIVERSE) == 29 and "DD" not in a.UNIVERSE and "RTX" in a.UNIVERSE and "UTX" not in a.UNIVERSE
    assert list(a.FEATURE_SETS) == ["A", "E"]
    assert a.FEATURE_SETS["A"]["n_features"] == 20 and a.FEATURE_SETS["E"]["n_features"] == 29
    e = a.FEATURE_SETS["E"]["columns"]
    assert {"stock_minus_spy_return_1", "stock_minus_spy_return_5"} <= set(e)
    assert not any("aapl" in c or c in ("symbol", "ticker") for fs in a.FEATURE_SETS.values() for c in fs["columns"])
    for fs in ("A", "E"):                                         # same definitions as Phase 10A, generic names only
        assert [p11d.COLUMN_RENAMES.get(c, c) for c in p10a.FEATURE_SETS[fs]["columns"]] == a.FEATURE_SETS[fs]["columns"]


def test_matrix_counts_and_families():
    m = a.matrix()
    assert m["counts"] == {"per_stock": 1914, "panel": 66, "total": 1980, "aapl_excluded_sensitivity": 66}
    assert m["families"]["panel"]["size"] == 66 and m["families"]["per_stock"]["size"] == 1914
    assert len(m["models"]) == 11
    d = p11d.design()["comparison_ids"]
    assert set(m["comparison_ids"]["per_stock"]) <= set(d["per_stock"])
    assert set(m["comparison_ids"]["panel"]) <= set(d["panel"])
    assert all(("_A_h" in i or "_E_h" in i) for i in m["comparison_ids"]["panel"])


def test_holm_family_and_qualification_rule():
    fam = [{"gate_status": QUALIFIED, "p_value_raw": 0.01}] + \
          [{"gate_status": "NOT_QUALIFIED", "p_value_raw": 0.5} for _ in range(65)]
    a.qualify_family(fam, 66)
    assert fam[0]["p_value_holm"] == pytest.approx(0.66) and fam[0]["qualified"] is False
    fam = [{"gate_status": QUALIFIED, "p_value_raw": 1e-5}] + fam[1:]
    a.qualify_family(fam, 66)
    assert fam[0]["qualified"] is True and not any(e["qualified"] for e in fam[1:])
    with pytest.raises(a.Phase11AError, match="Holm family"):
        a.qualify_family(fam[:10], 66)


# ---------------- panel folds ----------------


@pytest.mark.parametrize("h", [1, 3, 5])
def test_date_aligned_folds_with_gap(h):
    data = a.panel_rows(frames(), COLS, h)
    uniq = sorted(data["Date"].unique())
    pos = {d: i for i, d in enumerate(uniq)}
    for tr, te in a.date_splits(data["Date"], h, 5):
        assert not set(tr) & set(te)
        assert pos[te[0]] - pos[tr[-1]] == h + 1                  # exactly h dates skipped
        te_rows = data[data["Date"].isin(te)]
        assert len(te_rows) == len(te) * 3                        # every stock of a test date is in the test fold
    preds, folds = a.panel_walk_forward(data, COLS, FAST_PANEL, h, 5)
    for f in folds:
        assert f["train_end"] < f["test_start"]
    by_date = preds[preds["model"] == "Ridge"].groupby("Date")["fold"].nunique()
    assert (by_date == 1).all()


def test_no_ticker_feature_allowed():
    data = a.panel_rows(frames(), COLS, 1)
    with pytest.raises(a.Phase11AError, match="ticker"):
        a.panel_walk_forward(data, COLS + ["symbol"], FAST_PANEL, 1, 3)


def test_baselines_use_training_rows_only_and_per_stock():
    data = a.panel_rows(frames(), COLS, 1)
    preds, _ = a.panel_walk_forward(data, COLS, FAST_PANEL, 1, 4)
    for k, (tr, te) in enumerate(a.date_splits(data["Date"], 1, 4), start=1):
        train = data[data["Date"].isin(tr)]
        p = preds[preds["fold"] == k]
        mean = p[p["model"] == "Mean Return"]["prediction"]
        assert np.allclose(mean, train["future_return"].mean())
        rate = p[p["model"] == "Base Rate"]["prediction"]
        assert np.allclose(rate, train["direction"].mean())
        for sym, g in train.groupby("symbol"):
            ps = p[(p["model"] == a.PER_STOCK_MEAN.name) & (p["symbol"] == sym)]["prediction"]
            assert np.allclose(ps, g["future_return"].mean())
            pr = p[(p["model"] == a.PER_STOCK_RATE.name) & (p["symbol"] == sym)]["prediction"]
            assert np.allclose(pr, g["direction"].mean())
    # changing TEST-period targets never changes any baseline prediction of that fold
    tr, te = list(a.date_splits(data["Date"], 1, 4))[-1]
    data2 = data.copy()
    data2.loc[data2["Date"].isin(te), "future_return"] += 5.0
    data2.loc[data2["Date"].isin(te), "direction"] = 1
    p1 = preds[preds["fold"] == 4].sort_values(["model", "row"])
    p2, _ = a.panel_walk_forward(data2, COLS, FAST_PANEL, 1, 4)
    p2 = p2[p2["fold"] == 4].sort_values(["model", "row"])
    assert np.array_equal(p1["prediction"].to_numpy(), p2["prediction"].to_numpy())


# ---------------- per-date DM ----------------


@pytest.mark.parametrize("h", [1, 3])
def test_panel_dm_on_per_date_mean_loss(h):
    r = a.evaluate_panel(frames(), COLS, FAST_PANEL, horizon=h, n_splits=5)
    oos_dates = len({d for f in r["folds"] for d in pd.bdate_range(f["test_start"], f["test_end"])})
    for m in r["models"]:
        q = m["qualification"]
        if q["status"] == "BASELINE":
            continue
        assert q["n_dates"] == oos_dates and r["n_oos_rows"] == 3 * oos_dates   # dates, not stacked rows
        s = r["per_date_losses"][f"{m['task']}:{m['model']}"]
        ref = r["per_date_losses"][f"{m['task']}:{q['reference_baseline']}"]
        assert s["dates"] == ref["dates"]
        assert q["diebold_mariano"] == diebold_mariano(np.array(s["mean_loss"]), np.array(ref["mean_loss"]), h)


def test_per_date_mean_loss_construction():
    pred = pd.DataFrame({"Date": ["d1", "d1", "d2", "d2"], "y_true": [0.0, 0.0, 1.0, 1.0],
                         "prediction": [1.0, 3.0, 1.0, 0.0]})
    s = a._per_date_mean_loss(pred, "regression")
    assert s.tolist() == [5.0, 0.5]                                # mean of (1, 9) and (0, 1)


def test_one_stock_panel_equals_per_stock_harness():
    f = frame(7, n=220)
    for h in (1, 3):
        panel = a.evaluate_panel({"AAA": f}, COLS, FAST_PANEL, horizon=h, n_splits=6)
        stock = p10a.evaluate_excess(f, COLS, FAST, horizon=h, n_splits=6)
        pg = {m["model"]: m["qualification"] for m in panel["models"]}
        for m in stock["models"]:
            if m["is_baseline"]:
                continue
            q, p = m["qualification"], pg[m["model"]]
            assert p["status"] == q["status"]
            assert p["model_value"] == pytest.approx(q["model_value"], rel=1e-12)
            assert p["baseline_value"] == pytest.approx(q["baseline_value"], rel=1e-9)
            assert p["diebold_mariano"]["p_value"] == pytest.approx(q["diebold_mariano"]["p_value"], abs=1e-9)


# ---------------- frames: no future / confirmation data ----------------


def test_stock_frame_rejects_post_development_bars():
    late = pd.DataFrame({"Date": pd.to_datetime(["2024-09-30", "2024-10-01"]).astype("datetime64[ns]")})
    with pytest.raises(a.Phase11AError, match="cut"):
        a.stock_frame(late, late, late)
    assert a.dev_cut(late)["Date"].dt.date.max() <= a.DEV_END


def test_features_have_no_future_leakage(monkeypatch):
    monkeypatch.setattr(a, "DEV_START", pd.Timestamp("2017-02-01").date())
    full = a.stock_frame(bars(1, "2017-06-30"), bars(2, "2017-06-30"), bars(3, "2017-06-30"))
    cut = a.stock_frame(bars(1, "2017-03-31"), bars(2, "2017-03-31"), bars(3, "2017-03-31"))
    common = full[full["Date"] <= cut["Date"].iloc[-1]].reset_index(drop=True)
    pd.testing.assert_frame_equal(common, cut)                    # appending later bars changes no earlier row


# ---------------- AAPL reproducibility (amendment 01 tier 2) ----------------


def _repro_inputs(perturb=None):
    per_stock, r10a = [], {}
    for fs in ("A", "E"):
        for h in (1, 3, 5):
            for c in a.candidates():
                if c.is_baseline:
                    continue
                e = {"symbol": "AAPL", "feature_set": fs, "horizon": h, "model": c.name, "gate_status": "NOT_QUALIFIED",
                     "gate": {"model_value": 1.0}, "p_value_raw": 0.5}
                if perturb and perturb[0] == (fs, h, c.name):
                    e.update(perturb[1])
                per_stock.append(e)
                r10a[p10a.experiment_id(fs, h, c.name)] = {"development": {"status": "NOT_QUALIFIED",
                                                                           "gate": {"model_value": 1.002},
                                                                           "p_value": None, "p_value_raw": 0.51}}
    per_stock.append({**per_stock[0], "symbol": "AXP"})           # other stocks are ignored
    return per_stock, r10a


def test_aapl_reproducibility_tolerance():
    t1 = {"AAPL": {"pass": True}, "SPY": {"pass": True}, "QQQ": {"pass": True}}
    out = a.aapl_reproducibility(*_repro_inputs(), t1)
    assert out["tier2"]["n"] == 66 and out["outcome"] == "REPRODUCED_WITHIN_TOLERANCE" and out["failed"] == []
    for change in ({"gate_status": QUALIFIED}, {"gate": {"model_value": 1.01}}, {"p_value_raw": 0.54}):
        out = a.aapl_reproducibility(*_repro_inputs((("E", 3, "Ridge"), change)), t1)
        assert out["outcome"] == "REPRODUCIBILITY_CHECK_FAILED" and out["failed"] == ["E_h3_ridge"]
    per_stock, r10a = _repro_inputs()
    with pytest.raises(a.Phase11AError, match="66"):
        a.aapl_reproducibility(per_stock[:30], r10a, t1)


# ---------------- registry ----------------


def test_registry_write_once_and_tamper(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "verified_inputs", lambda: {"artifacts": {}})
    path = tmp_path / "r.json"
    reg = a.freeze(path)
    assert reg["status"] == "FROZEN" and reg["confirmation"] == "NOT_EVALUATED"
    assert a.load_registry(path)["matrix_sha256"] == reg["matrix_sha256"]
    with pytest.raises(a.Phase11AError, match="frozen once"):
        a.freeze(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    d["matrix"]["horizons"] = [1]
    path.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(a.Phase11AError, match="edited"):
        a.load_registry(path)


# ---------------- frozen inputs (real files) ----------------


def test_canonical_inputs_and_frozen_artifacts_unchanged():
    if not a.PRICE_REPORT.exists():
        pytest.skip("Phase 11.1 artifacts not present")
    inp = a.verified_inputs()
    assert set(inp["artifacts"]) == set(a.UNIVERSE) | {"SPY", "QQQ"}
    assert inp["artifacts"]["AAPL"]["sha256"] == "e4f48991f876d8e85fe85fc434065603dd1f2e4e4db2c69f45082a25d4f0cde8"
    assert inp["design_sha256"] == p11d.load_registry()["design_sha256"]
