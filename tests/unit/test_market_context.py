"""market_context_v1: formulas, timestamps, strict calendar alignment, no future data, determinism."""

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from services.market_calendar_service import TradingCalendar
from services.market_context import (
    COLUMNS,
    MAX_LOOKBACK,
    MarketContextError,
    compute_market_context,
    load_context_snapshot,
)
from services.news_alignment import row_prediction_timestamp
from training.build_dataset import save_raw_snapshot

HOLIDAYS = {date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29), date(2024, 5, 27), date(2024, 6, 19)}
SESSIONS = [d for d in pd.bdate_range("2024-01-02", "2024-06-28").date if d not in HOLIDAYS]


def bars(dates, closes):
    closes = np.asarray(closes, dtype="float64")
    return pd.DataFrame({"Date": pd.to_datetime(list(dates)).astype("datetime64[ns]"), "Open": closes,
                         "High": closes * 1.01, "Low": closes * 0.99, "Close": closes,
                         "Volume": np.full(len(closes), 1_000_000.0)})


def walk(seed, n=len(SESSIONS)):
    return 100 * np.cumprod(1 + np.random.default_rng(seed).normal(0, 0.01, n))


A, S, Q = walk(1), walk(2), walk(3)
ROWS = SESSIONS[MAX_LOOKBACK:]                                    # first row with a full 50-session history


def compute(a=A, s=S, q=Q, dates=ROWS, s_dates=SESSIONS, q_dates=SESSIONS):
    return compute_market_context(bars(SESSIONS, a), bars(s_dates, s), bars(q_dates, q), dates)


def test_formulas_match_hand_computation():
    out = compute().set_index("trading_date")
    for i in (MAX_LOOKBACK, 70, len(SESSIONS) - 1):
        r = out.loc[SESSIONS[i]]
        assert r["spy_return_1"] == pytest.approx(S[i] / S[i - 1] - 1, rel=1e-12)
        assert r["spy_return_5"] == pytest.approx(S[i] / S[i - 5] - 1, rel=1e-12)
        assert r["spy_return_20"] == pytest.approx(S[i] / S[i - 20] - 1, rel=1e-12)
        assert r["qqq_return_1"] == pytest.approx(Q[i] / Q[i - 1] - 1, rel=1e-12)
        assert r["qqq_return_5"] == pytest.approx(Q[i] / Q[i - 5] - 1, rel=1e-12)
        for k in (1, 5):
            assert r[f"aapl_minus_spy_return_{k}"] == pytest.approx(
                (A[i] / A[i - k] - 1) - (S[i] / S[i - k] - 1), rel=1e-9, abs=1e-15)
        logret = np.log(S[i - 19:i + 1] / S[i - 20:i])
        assert len(logret) == 20
        assert r["spy_volatility_20"] == pytest.approx(np.std(logret, ddof=1), rel=1e-9)
        assert r["spy_close_to_sma_50"] == pytest.approx(S[i] / S[i - 49:i + 1].mean(), rel=1e-12)


def test_prediction_timestamp_is_d_1630_new_york():
    out = compute()
    cal = TradingCalendar(SESSIONS)
    assert all(t == row_prediction_timestamp(d, cal) for d, t in zip(out["trading_date"], out["prediction_timestamp"]))
    # bar D is complete at D 16:30 New York (EDT in April -> 20:30 UTC) = the prediction timestamp
    assert out.set_index("trading_date").loc[date(2024, 4, 15), "prediction_timestamp"] == \
        datetime(2024, 4, 15, 20, 30, tzinfo=timezone.utc)


def test_no_future_data_is_used():
    cut = 80
    base = compute().set_index("trading_date")
    a2, s2, q2 = A.copy(), S.copy(), Q.copy()
    for arr in (a2, s2, q2):
        arr[cut + 1:] *= 1.5                                      # change every bar AFTER row `cut`
    changed = compute(a2, s2, q2).set_index("trading_date")
    upto = [d for d in ROWS if d <= SESSIONS[cut]]
    pd.testing.assert_frame_equal(base.loc[upto], changed.loc[upto])
    assert not base.loc[SESSIONS[cut + 1]].equals(changed.loc[SESSIONS[cut + 1]])


def test_missing_context_session_raises_no_forward_fill():
    gap = SESSIONS[60]
    with pytest.raises(MarketContextError, match="SPY: 1 AAPL session"):
        compute(s=np.delete(S, 60), s_dates=[d for d in SESSIONS if d != gap])
    with pytest.raises(MarketContextError, match="QQQ: 1 AAPL session"):
        compute(q=np.delete(Q, 60), q_dates=[d for d in SESSIONS if d != gap])


def test_missing_session_inside_the_lookback_also_raises():
    gap = SESSIONS[MAX_LOOKBACK - 10]                              # before the first row, inside its SMA window
    with pytest.raises(MarketContextError, match="missing"):
        compute(s=np.delete(S, MAX_LOOKBACK - 10), s_dates=[d for d in SESSIONS if d != gap])


def test_extra_context_session_raises():
    extra = date(2024, 2, 19)                                     # AAPL holiday
    dates = sorted(SESSIONS + [extra])
    with pytest.raises(MarketContextError, match="not AAPL sessions"):
        compute(s=walk(9, len(dates)), s_dates=dates)


def test_insufficient_history_raises():
    with pytest.raises(MarketContextError, match="fewer than"):
        compute(dates=[SESSIONS[MAX_LOOKBACK - 1]])


def test_non_session_or_duplicate_dates_raise():
    with pytest.raises(MarketContextError):
        compute(dates=[date(2024, 5, 27)])                        # holiday
    with pytest.raises(MarketContextError):
        compute(dates=[ROWS[0], ROWS[0]])


def test_deterministic_and_complete():
    a, b = compute(), compute()
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["trading_date", "prediction_timestamp", *COLUMNS]
    assert len(a) == len(ROWS) and np.isfinite(a[list(COLUMNS)].to_numpy()).all()


def test_snapshot_ticker_is_checked(tmp_path):
    path = save_raw_snapshot(bars(SESSIONS, Q), "QQQ", "1y", datetime(2024, 7, 1, tzinfo=timezone.utc), raw_dir=tmp_path)
    loaded, meta = load_context_snapshot(path, "QQQ")
    assert meta["ticker"] == "QQQ" and len(loaded) == len(SESSIONS)
    with pytest.raises(MarketContextError, match="expected 'SPY'"):
        load_context_snapshot(path, "SPY")
