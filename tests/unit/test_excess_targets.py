"""Phase 10A excess-return targets: formula, direction, horizons, availability, leakage."""

import numpy as np
import pandas as pd
import pytest

from training.excess_targets import (
    HORIZONS,
    SPY_CLOSE,
    TARGET_COLUMNS,
    ExcessTargetError,
    build_excess_targets,
    future_excess_return,
    target_definition,
)
from training.phase10a_research import FEATURE_SETS

AAPL = np.array([100.0, 101.0, 99.5, 102.0, 103.0, 101.0, 104.0, 104.5, 103.0, 106.0, 107.0, 105.5])
SPY = np.array([400.0, 401.0, 402.0, 400.5, 403.0, 404.0, 403.5, 405.0, 406.0, 405.5, 407.0, 408.0])


def frame(aapl=AAPL, spy=SPY):
    return pd.DataFrame({"Date": pd.bdate_range("2024-01-02", periods=len(aapl)), "Close": aapl, SPY_CLOSE: spy})


@pytest.mark.parametrize("h", HORIZONS)
def test_excess_formula_and_legs(h):
    t = build_excess_targets(frame(), h)
    a = AAPL[h:] / AAPL[:-h] - 1
    s = SPY[h:] / SPY[:-h] - 1
    assert len(t) == len(AAPL) - h                                    # last h rows have no label
    np.testing.assert_array_equal(t["aapl_future_return"].to_numpy(), a)
    np.testing.assert_array_equal(t["spy_future_return"].to_numpy(), s)
    np.testing.assert_array_equal(t["future_return"].to_numpy(), a - s)
    assert (t["direction"] == ((a - s) > 0).astype(int)).all()


@pytest.mark.parametrize("h", HORIZONS)
def test_future_excess_return_function_matches(h):
    f = frame()
    got = future_excess_return(f["Close"], f[SPY_CLOSE], h)
    assert got.iloc[-h:].isna().all()
    np.testing.assert_array_equal(got.iloc[:-h].to_numpy(), build_excess_targets(f, h)["future_return"].to_numpy())


def test_zero_excess_is_down():
    same = np.array([100.0, 101.0, 102.0, 101.0])
    t = build_excess_targets(frame(same, same * 4), 1)                # identical returns -> excess exactly 0
    assert (t["future_return"] == 0).all() and (t["direction"] == 0).all()


@pytest.mark.parametrize("h", HORIZONS)
def test_label_uses_only_closes_t_and_t_plus_h(h):
    k = 5
    base = build_excess_targets(frame(), h)
    a2, s2 = AAPL.copy(), SPY.copy()
    a2[k + h + 1:] *= 2.0                                               # change closes after t+h for rows <= k
    s2[k + h + 1:] *= 0.5
    changed = build_excess_targets(frame(a2, s2), h)
    pd.testing.assert_frame_equal(base.iloc[:k + 1], changed.iloc[:k + 1])
    assert not np.isclose(base["future_return"].iloc[k + 1], changed["future_return"].iloc[k + 1])


def test_target_definition_contract():
    for h in HORIZONS:
        d = target_definition(h)
        assert d["regression"]["name"] == f"future_excess_return_{h}d"
        assert d["classification"]["name"] == f"excess_direction_{h}d"
        assert d["walk_forward_gap"] == h and d["last_rows_without_label"] == h
        assert "never a feature" in d["future_spy_use"]


def test_invalid_inputs_raise():
    with pytest.raises(ExcessTargetError):
        build_excess_targets(frame(), 2)                                 # not a Phase 10A horizon
    with pytest.raises(ExcessTargetError):
        build_excess_targets(frame().drop(columns=[SPY_CLOSE]), 1)
    bad = SPY.copy()
    bad[3] = np.nan
    with pytest.raises(ExcessTargetError):
        build_excess_targets(frame(spy=bad), 1)
    with pytest.raises(ExcessTargetError):
        build_excess_targets(frame(AAPL[:3], SPY[:3]), 5)


def test_no_target_or_price_column_is_a_feature():
    for fs in FEATURE_SETS.values():
        assert not set(fs["columns"]) & set(TARGET_COLUMNS)
