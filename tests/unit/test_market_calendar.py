"""Completed-bar rule (services/market_calendar_service.py)."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import services.live_stock_service as live
from services.market_calendar_service import (
    completion_time,
    drop_incomplete_bars,
    is_bar_complete,
)
from services.market_data import MarketDataError
from tests.conftest import as_ticker_history, make_bars

NY = ZoneInfo("America/New_York")


def ny(*args):
    return datetime(*args, tzinfo=NY)


# ==========================================
# The rule: complete iff now >= D 16:00 New York + 30 min
# ==========================================


@pytest.mark.parametrize("bar_date, now, expected", [
    # summer (EDT, UTC-4)
    ("2024-07-10", ny(2024, 7, 10, 10, 0), False),       # session in progress
    ("2024-07-10", ny(2024, 7, 10, 16, 0), False),       # at the bell, not settled
    ("2024-07-10", ny(2024, 7, 10, 16, 29, 59), False),  # 1s before cutoff
    ("2024-07-10", ny(2024, 7, 10, 16, 30), True),       # exactly at cutoff
    ("2024-07-10", ny(2024, 7, 11, 9, 0), True),         # next morning
    ("2024-07-09", ny(2024, 7, 10, 10, 0), True),        # earlier session
    # winter (EST, UTC-5)
    ("2024-01-16", datetime(2024, 1, 16, 21, 29, tzinfo=timezone.utc), False),  # 16:29 EST
    ("2024-01-16", datetime(2024, 1, 16, 21, 30, tzinfo=timezone.utc), True),   # 16:30 EST
])
def test_completed_bar_rule(bar_date, now, expected):
    assert is_bar_complete(bar_date, now) is expected


def test_completion_time_handles_dst():
    assert completion_time("2024-07-10").astimezone(timezone.utc).hour == 20  # 16:30 EDT
    assert completion_time("2024-01-16").astimezone(timezone.utc).hour == 21  # 16:30 EST


def test_future_bar_is_an_error():
    with pytest.raises(MarketDataError, match="future"):
        is_bar_complete("2024-07-11", ny(2024, 7, 10, 18, 0))


def test_naive_now_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        is_bar_complete("2024-07-10", datetime(2024, 7, 11, 12, 0))


# ==========================================
# drop_incomplete_bars
# ==========================================


def test_in_progress_bar_is_dropped():
    bars = make_bars(10, start="2024-07-01")         # last bar: 2024-07-12 (Fri)
    last = bars["Date"].iloc[-1]
    during_session = ny(2024, 7, 12, 11, 0)
    out = drop_incomplete_bars(bars, during_session)
    assert len(out) == 9
    assert out["Date"].iloc[-1] < last
    assert out.attrs["incomplete_bars_dropped"] == 1


def test_completed_last_bar_is_kept():
    bars = make_bars(10, start="2024-07-01")
    out = drop_incomplete_bars(bars, ny(2024, 7, 12, 17, 0))
    assert len(out) == 10
    assert out.attrs["incomplete_bars_dropped"] == 0


def test_only_bar_incomplete_raises():
    bars = make_bars(1, start="2024-07-12")
    with pytest.raises(MarketDataError, match="No completed"):
        drop_incomplete_bars(bars, ny(2024, 7, 12, 11, 0))


def test_fetch_drops_intraday_bar(monkeypatch):
    bars = make_bars(120, start="2024-02-01")
    session_day = bars["Date"].iloc[-1]
    monkeypatch.setattr(live, "_fetch_history", lambda s, p: as_ticker_history(bars))
    now = datetime.combine(session_day.date(), datetime.min.time(), tzinfo=NY).replace(hour=12)

    out = live.fetch_latest_stock_data("AAPL", now=now, retry_delay=0)
    assert out["Date"].iloc[-1] == bars["Date"].iloc[-2]

    kept = live.fetch_latest_stock_data("AAPL", now=now, completed_only=False, retry_delay=0)
    assert kept["Date"].iloc[-1] == session_day
    assert isinstance(session_day, pd.Timestamp)
