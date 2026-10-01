"""Phase 10B normalized targets: arithmetic, direction, volatility window, cutoff, leakage, history."""

import numpy as np
import pandas as pd
import pytest

from training.excess_targets import SPY_CLOSE, build_excess_targets
from training.normalized_targets import (
    NORMALIZED_TARGET_COLUMNS,
    VOL_COLUMN,
    VOL_WINDOW,
    NormalizedTargetError,
    attach_volatility,
    build_normalized_targets,
    daily_excess_returns,
    prediction_time_volatility,
    target_definition,
)
from training.phase10b_research import FEATURE_SETS

N = 80
DATES = pd.bdate_range("2024-01-02", periods=N)
rng = np.random.default_rng(3)
A = 100 * np.cumprod(1 + rng.normal(0, 0.015, N))
S = 400 * np.cumprod(1 + rng.normal(0, 0.010, N))


def bars(dates, close):
    close = np.asarray(close, dtype="float64")
    return pd.DataFrame({"Date": pd.to_datetime(list(dates)).astype("datetime64[ns]"), "Open": close,
                         "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1e6})


def frame_from(start=30, a=A, s=S):
    """Frame rows = sessions start.. (earlier sessions only exist in the 'snapshots')."""
    f = pd.DataFrame({"Date": DATES[start:], "Close": a[start:], SPY_CLOSE: s[start:]}).reset_index(drop=True)
    return attach_volatility(f, bars(DATES, a), bars(DATES, s))


def test_volatility_is_trailing_20_session_sample_std_of_daily_excess_returns():
    f = frame_from()
    for i in (0, 10, len(f) - 1):
        d = 30 + i                                                 # snapshot position of row i
        x = (A[d - 19:d + 1] / A[d - 20:d] - 1) - (S[d - 19:d + 1] / S[d - 20:d] - 1)
        assert len(x) == VOL_WINDOW
        assert f[VOL_COLUMN].iloc[i] == pytest.approx(np.std(x, ddof=1), rel=1e-10)


def test_volatility_window_boundaries():
    x = daily_excess_returns(pd.Series(A), pd.Series(S))
    vol = prediction_time_volatility(pd.Series(A), pd.Series(S))
    assert vol.iloc[:VOL_WINDOW].isna().all()                    # 20 returns need 21 closes (index 20)
    assert np.isfinite(vol.iloc[VOL_WINDOW])
    assert vol.iloc[VOL_WINDOW] == pytest.approx(x.iloc[1:VOL_WINDOW + 1].std(ddof=1), rel=1e-12)


def test_future_mutation_does_not_change_prediction_time_volatility():
    base = frame_from()
    a2, s2 = A.copy(), S.copy()
    k = 50                                                         # snapshot position
    a2[k + 1:] *= 1.7
    s2[k + 1:] *= 0.6
    changed = frame_from(a=a2, s=s2)
    upto = base["Date"] <= DATES[k]
    np.testing.assert_array_equal(base.loc[upto, VOL_COLUMN].to_numpy(), changed.loc[upto, VOL_COLUMN].to_numpy())
    assert not np.isclose(base.loc[~upto, VOL_COLUMN].iloc[0], changed.loc[~upto, VOL_COLUMN].iloc[0])


@pytest.mark.parametrize("h", [1, 3, 5])
def test_normalized_target_arithmetic_and_direction(h):
    f = frame_from()
    t = build_normalized_targets(f, h)
    e = build_excess_targets(f[["Date", "Close", SPY_CLOSE]], h)["future_return"].to_numpy()
    v = f[VOL_COLUMN].to_numpy()[: len(f) - h]
    assert len(t) == len(f) - h
    np.testing.assert_array_equal(t["future_excess_return"].to_numpy(), e)
    np.testing.assert_array_equal(t["future_return"].to_numpy(), e / v)
    assert (t["direction"] == (e / v > 0).astype(int)).all()
    assert (t["direction"] == (e > 0).astype(int)).all()          # vol > 0: same sign as the excess return


def test_target_uses_volatility_at_d_not_later():
    f = frame_from()
    g = f.copy()
    g.loc[g.index[-1], VOL_COLUMN] *= 10.0                         # change volatility of the LAST row only
    t1, t2 = build_normalized_targets(f, 1), build_normalized_targets(g, 1)
    pd.testing.assert_frame_equal(t1, t2)                          # last row has no label; earlier rows unaffected


def test_insufficient_history_and_bad_volatility_raise():
    with pytest.raises(NormalizedTargetError, match="insufficient history"):
        frame_from(start=10)                                       # only 10 returns before the first row
    f = frame_from()
    g = f.copy()
    g.loc[3, VOL_COLUMN] = 0.0
    with pytest.raises(NormalizedTargetError):
        build_normalized_targets(g, 1)
    with pytest.raises(NormalizedTargetError):
        build_normalized_targets(f.drop(columns=[VOL_COLUMN]), 1)


def test_missing_spy_session_raises():
    f = pd.DataFrame({"Date": DATES[30:], "Close": A[30:], SPY_CLOSE: S[30:]})
    spy = bars(DATES, S).drop(index=25)                            # inside the 20-return lookback of row 0
    with pytest.raises(NormalizedTargetError, match="SPY close missing"):
        attach_volatility(f, bars(DATES, A), spy)


def test_definition_and_no_target_columns_as_features():
    for h in (1, 3, 5):
        d = target_definition(h)
        assert d["regression"]["name"] == f"normalized_excess_return_{h}d"
        assert d["classification"]["name"] == f"normalized_excess_direction_{h}d"
        assert d["volatility"]["window_sessions"] == 20 and d["volatility"]["ddof"] == 1
        assert d["walk_forward_gap"] == h
    for fs in FEATURE_SETS.values():
        assert not set(fs["columns"]) & set(NORMALIZED_TARGET_COLUMNS)
