"""Phase 10C regimes: prediction-time safety, fold-fitted thresholds, labels (leakage items 1-9)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from services.market_context import compute_market_context
from training import regimes as rg
from training.excess_targets import TARGET_COLUMNS
from training.phase10c_research import FEATURE_SETS

N = 120
DATES = pd.bdate_range("2024-01-02", periods=N)
rng = np.random.default_rng(5)
A = 100 * np.cumprod(1 + rng.normal(0, 0.015, N))
S = 400 * np.cumprod(1 + rng.normal(0, 0.010, N))
Q = 300 * np.cumprod(1 + rng.normal(0, 0.012, N))


def bars(close):
    close = np.asarray(close, dtype="float64")
    return pd.DataFrame({"Date": pd.to_datetime(list(DATES)).astype("datetime64[ns]"), "Open": close,
                         "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1e6})


def frame(a=A, s=S, start=60):
    mc = compute_market_context(bars(a), bars(s), bars(Q), list(DATES[start:].date))
    f = pd.DataFrame({"Date": DATES[start:]}).reset_index(drop=True)
    f["Close"] = a[start:]
    f["spy_close_to_sma_50"] = mc["spy_close_to_sma_50"].to_numpy()
    f["spy_return_20"] = mc["spy_return_20"].to_numpy()
    return rg.attach_regime_variables(f, bars(a), bars(s))


def values(f):
    return {c: f[c].to_numpy(dtype="float64") for c in rg.REGIME_INPUT_COLUMNS}


def test_realized_volatility_formula_and_window():
    v = rg.realized_volatility(pd.Series(A))
    assert v.iloc[:20].isna().all() and np.isfinite(v.iloc[20])
    r = A[1:21] / A[0:20] - 1
    assert v.iloc[20] == pytest.approx(np.std(r, ddof=1), rel=1e-12)
    f = frame()
    i = 70                                                         # snapshot position of a frame row
    r = A[i - 19:i + 1] / A[i - 20:i] - 1
    assert f.loc[f["Date"] == DATES[i], rg.AAPL_VOL].iloc[0] == pytest.approx(np.std(r, ddof=1), rel=1e-10)


@pytest.mark.parametrize("which", ["aapl", "spy"])
def test_future_prices_do_not_change_regime_variables_at_d(which):
    k = 80
    a2, s2 = A.copy(), S.copy()
    (a2 if which == "aapl" else s2)[k + 1:] *= 1.9
    base, changed = frame(), frame(a2, s2)
    upto = base["Date"] <= DATES[k]
    for c in rg.REGIME_INPUT_COLUMNS:                              # vol (1-3), trend (4), 20-day return (5)
        np.testing.assert_array_equal(base.loc[upto, c].to_numpy(), changed.loc[upto, c].to_numpy())
    col = rg.AAPL_VOL if which == "aapl" else rg.SPY_VOL
    assert not np.isclose(base.loc[~upto, col].iloc[0], changed.loc[~upto, col].iloc[0])


def test_threshold_uses_training_rows_only():
    v = values(frame())
    train, test = np.arange(0, 30), np.arange(32, 50)
    t1 = rg.fold_thresholds(v, train)
    w = {k: x.copy() for k, x in v.items()}
    w[rg.AAPL_VOL][test] *= 100.0                                  # validation rows: no effect
    w[rg.SPY_VOL][test] *= 100.0
    assert rg.fold_thresholds(w, train) == t1
    assert t1["R1_aapl_vol"] == pytest.approx(np.median(v[rg.AAPL_VOL][train]))
    w[rg.AAPL_VOL][train[:20]] *= 100.0                            # training rows: threshold moves
    assert rg.fold_thresholds(w, train)["R1_aapl_vol"] != t1["R1_aapl_vol"]


def test_labels_use_frozen_threshold_and_rules():
    v = {rg.AAPL_VOL: np.array([1.0, 2.0, 3.0]), rg.SPY_VOL: np.array([0.5, 0.5, 0.9]),
         rg.SPY_TREND: np.array([1.0, 0.99, 1.2]), rg.SPY_RET20: np.array([0.0, -0.01, 0.02])}
    lab = rg.label_rows(v, np.arange(3), {"R1_aapl_vol": 2.0, "R2_spy_vol": 0.5})
    assert list(lab["R1_aapl_vol"]) == ["LOW_VOL", "LOW_VOL", "HIGH_VOL"]          # tie == median -> LOW
    assert list(lab["R2_spy_vol"]) == ["LOW_SPY_VOL", "LOW_SPY_VOL", "HIGH_SPY_VOL"]
    assert list(lab["R3_spy_trend"]) == ["POSITIVE_TREND", "NEGATIVE_TREND", "POSITIVE_TREND"]   # >= 1.0
    assert list(lab["R4_spy_return_20"]) == ["POSITIVE_20D_RETURN", "NEGATIVE_20D_RETURN", "POSITIVE_20D_RETURN"]
    assert list(lab["R5_spy_vol_x_trend"]) == ["LOW_SPY_VOL_POSITIVE_TREND", "LOW_SPY_VOL_NEGATIVE_TREND",
                                               "HIGH_SPY_VOL_POSITIVE_TREND"]


def test_assign_oos_states_per_fold():
    v = values(frame())
    splits = [(np.arange(0, 20), np.arange(21, 35)), (np.arange(0, 35), np.arange(36, 50))]
    states, folds = rg.assign_oos_states(v, splits)
    assert set(states["R1_aapl_vol"]) == set(range(21, 35)) | set(range(36, 50))
    for f, (train, test) in zip(folds, splits):
        thr = f["thresholds"]["R1_aapl_vol"]
        assert thr == pytest.approx(np.median(v[rg.AAPL_VOL][train]))
        for r in test:
            assert states["R1_aapl_vol"][int(r)] == ("LOW_VOL" if v[rg.AAPL_VOL][r] <= thr else "HIGH_VOL")
        assert sum(f["state_counts"]["R5_spy_vol_x_trend"].values()) == len(test)


def test_insufficient_history_raises():
    with pytest.raises(rg.RegimeError, match="insufficient history"):
        rg.attach_regime_variables(pd.DataFrame({"Date": DATES[10:]}), bars(A), bars(S))


def test_no_target_columns_in_regime_inputs_and_no_regime_variables_as_features():
    assert not set(rg.REGIME_INPUT_COLUMNS) & set(TARGET_COLUMNS)
    for fs in FEATURE_SETS.values():
        assert not set(fs["columns"]) & set(rg.REGIME_VARIABLE_COLUMNS)
    assert rg.STATE_COUNT == 12
