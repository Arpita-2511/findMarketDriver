"""TradingCalendar (bar-derived trading sessions) - Phase 4B."""

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from services.market_calendar_service import (
    OutsideCalendarError,
    TradingCalendar,
    completion_time,
)
from services.market_data import MarketDataError
from tests.conftest import calendar_2024, make_bars

NY = ZoneInfo("America/New_York")


def test_from_bars_uses_bar_dates_as_sessions():
    bars = make_bars(30, start="2024-07-01")
    cal = TradingCalendar.from_bars(bars)
    assert cal.first_session == date(2024, 7, 1)
    assert cal.last_session == bars["Date"].iloc[-1].date()


def test_holidays_and_weekends_are_not_sessions():
    cal = calendar_2024()
    assert cal.is_session(date(2024, 7, 3))
    assert not cal.is_session(date(2024, 7, 4))          # holiday: no bar
    assert not cal.is_session(date(2024, 7, 6))          # Saturday


@pytest.mark.parametrize("day, expected", [
    (date(2024, 7, 3), date(2024, 7, 3)),                # session itself
    (date(2024, 7, 4), date(2024, 7, 5)),                # holiday -> next
    (date(2024, 7, 6), date(2024, 7, 8)),                # Saturday -> Monday
    (date(2024, 5, 25), date(2024, 5, 28)),              # weekend + Memorial Day -> Tuesday
])
def test_next_session_on_or_after(day, expected):
    assert calendar_2024().next_session_on_or_after(day) == expected


def test_next_session_after():
    cal = calendar_2024()
    assert cal.next_session_after(date(2024, 7, 12)) == date(2024, 7, 15)   # Friday -> Monday
    assert cal.next_session_after(date(2024, 7, 3)) == date(2024, 7, 5)     # skips July 4


@pytest.mark.parametrize("day", [date(2024, 2, 29), date(2024, 8, 1)])
def test_outside_known_range_is_an_error_not_a_guess(day):
    cal = calendar_2024()
    with pytest.raises(OutsideCalendarError):
        cal.is_session(day)
    with pytest.raises(OutsideCalendarError):
        cal.next_session_after(date(2024, 7, 31))


def test_invalid_calendars_rejected():
    with pytest.raises(MarketDataError, match="at least one"):
        TradingCalendar([])
    with pytest.raises(MarketDataError, match="Weekend"):
        TradingCalendar([date(2024, 7, 6)])


def test_date_inputs():
    cal = calendar_2024()
    assert cal.is_session(pd.Timestamp("2024-07-10"))    # canonical bar Date
    assert cal.is_session("2024-07-10")
    with pytest.raises(TypeError):
        cal.is_session(datetime(2024, 7, 10, 10, 30))
    with pytest.raises(TypeError):
        cal.is_session(datetime(2024, 7, 10, tzinfo=timezone.utc))


def test_session_open_close_follow_dst():
    cal = calendar_2024()
    assert cal.session_open(date(2024, 3, 8)) == datetime(2024, 3, 8, 14, 30, tzinfo=timezone.utc)   # EST
    assert cal.session_open(date(2024, 3, 11)) == datetime(2024, 3, 11, 13, 30, tzinfo=timezone.utc)  # EDT
    assert cal.session_close(date(2024, 7, 10)) == datetime(2024, 7, 10, 20, 0, tzinfo=timezone.utc)
    with pytest.raises(MarketDataError, match="not a trading session"):
        cal.session_open(date(2024, 7, 4))


def test_phase2_completed_bar_rule_unchanged():
    assert completion_time("2024-07-10") == datetime(2024, 7, 10, 16, 30, tzinfo=NY)
