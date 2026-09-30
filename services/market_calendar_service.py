"""
Completed-bar policy for daily bars (Phase 2).

yfinance returns the CURRENT session as a daily bar while the market is
open - its "Close" is just the latest trade. Such a bar must never be
treated as a completed daily observation (REQ-DATA-002).

Rule
----
A daily bar dated D (exchange-local date, America/New_York) is COMPLETE
at time `now` if and only if

    now >= D 16:00 America/New_York + SETTLEMENT_DELAY (30 minutes)

- 16:00 is the regular NYSE/Nasdaq close. zoneinfo handles EST/EDT.
- The 30-minute delay gives the data vendor time to publish the final
  close/volume for the day.
- Bars dated before today (exchange time) are therefore always complete.
- A bar dated AFTER today (exchange time) is impossible and is an error.

Known limitation: early-close days (e.g. 13:00 closes) are treated
conservatively - their bar is considered complete only from 16:30, so
a run between 13:00 and 16:30 on such a day uses the previous session.
No holiday calendar is needed: holidays and weekends produce no bar.

Trading sessions (Phase 4B)
---------------------------
TradingCalendar applies the same principle: the trading sessions ARE the
dates of completed daily bars. A weekday inside the calendar's range
without a bar is a non-trading day (holiday). Dates outside the range
are unknown and raise OutsideCalendarError instead of being guessed.
"""

from __future__ import annotations

import bisect
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from services.market_data import EXCHANGE_TIMEZONE, MarketDataError, validate_ohlcv

MARKET_TZ = ZoneInfo(EXCHANGE_TIMEZONE)
REGULAR_SESSION_OPEN = time(9, 30)
REGULAR_SESSION_CLOSE = time(16, 0)
SETTLEMENT_DELAY = timedelta(minutes=30)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def completion_time(bar_date) -> datetime:
    """Earliest moment (timezone-aware) at which the bar for `bar_date` counts as complete."""
    d = pd.Timestamp(bar_date).date()
    return datetime.combine(d, REGULAR_SESSION_CLOSE, tzinfo=MARKET_TZ) + SETTLEMENT_DELAY


def _require_aware(now: datetime) -> datetime:
    if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
        raise ValueError("`now` must be timezone-aware")
    return now


def is_bar_complete(bar_date, now: datetime) -> bool:
    """Apply the completed-bar rule. Raises MarketDataError for future-dated bars."""
    now = _require_aware(now)
    today_exchange = now.astimezone(MARKET_TZ).date()
    d = pd.Timestamp(bar_date).date()
    if d > today_exchange:
        raise MarketDataError(f"Bar dated {d} is in the future (exchange date is {today_exchange})")
    return now >= completion_time(d)


def drop_incomplete_bars(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """
    Return only completed bars. `df` must already be in the canonical
    schema, sorted by Date. Only the most recent bar can legitimately be
    incomplete; an incomplete bar anywhere else is reported as an error.
    The number of dropped bars is recorded in df.attrs["incomplete_bars_dropped"].
    """
    now = _require_aware(now or utc_now())
    complete = df["Date"].map(lambda d: is_bar_complete(d, now))

    n_incomplete = int((~complete).sum())
    if n_incomplete > 1 or (n_incomplete == 1 and complete.iloc[-1]):
        raise MarketDataError("Incomplete bar found before the most recent bar")

    out = df[complete].reset_index(drop=True)
    out.attrs = {**df.attrs, "incomplete_bars_dropped": n_incomplete, "completed_as_of": now.isoformat()}
    if out.empty:
        raise MarketDataError("No completed daily bars available")
    return out


# ==========================================
# Trading sessions (Phase 4B)
# ==========================================


class OutsideCalendarError(MarketDataError):
    """A date falls outside the range of known trading sessions."""


def _as_date(value) -> date:
    """
    Accept a calendar date: date, 'YYYY-MM-DD', or a naive midnight
    Timestamp (the canonical bar Date). A datetime with a time of day or a
    timezone is rejected - convert it to an exchange-local date explicitly.
    """
    if isinstance(value, datetime):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None and ts == ts.normalize():
            return ts.date()
        raise TypeError("expected a calendar date, got a datetime with time/timezone "
                        "(convert to an exchange-local date first)")
    return value if isinstance(value, date) else pd.Timestamp(value).date()


class TradingCalendar:
    """
    Trading sessions derived from completed daily bars (no external
    holiday list). Sessions are exchange-local (America/New_York) dates.
    """

    def __init__(self, sessions: Iterable):
        days = sorted({_as_date(d) for d in sessions})
        if not days:
            raise MarketDataError("TradingCalendar needs at least one session")
        weekend = [d for d in days if d.weekday() >= 5]
        if weekend:
            raise MarketDataError(f"Weekend dates are not trading sessions: {weekend[:3]}")
        self._sessions = days
        self._session_set = set(days)

    @classmethod
    def from_bars(cls, bars: pd.DataFrame) -> "TradingCalendar":
        """Build from canonical OHLCV bars (services/market_data.py), e.g. a raw snapshot."""
        validate_ohlcv(bars)
        return cls(bars["Date"].dt.date)

    @property
    def first_session(self) -> date:
        return self._sessions[0]

    @property
    def last_session(self) -> date:
        return self._sessions[-1]

    def _check_range(self, d: date) -> None:
        if d < self.first_session or d > self.last_session:
            raise OutsideCalendarError(
                f"{d} is outside the known sessions {self.first_session} .. {self.last_session}"
            )

    def is_session(self, d) -> bool:
        d = _as_date(d)
        self._check_range(d)
        return d in self._session_set

    def next_session_on_or_after(self, d) -> date:
        d = _as_date(d)
        if d < self.first_session:
            raise OutsideCalendarError(f"{d} is before the first known session {self.first_session}")
        i = bisect.bisect_left(self._sessions, d)
        if i == len(self._sessions):
            raise OutsideCalendarError(
                f"No known session on or after {d}; the last known session is {self.last_session}"
            )
        return self._sessions[i]

    def next_session_after(self, d) -> date:
        return self.next_session_on_or_after(_as_date(d) + timedelta(days=1))

    def session_open(self, d) -> datetime:
        """Regular-session open (09:30 New York) as a UTC datetime."""
        d = self._require_session(d)
        return datetime.combine(d, REGULAR_SESSION_OPEN, tzinfo=MARKET_TZ).astimezone(timezone.utc)

    def session_close(self, d) -> datetime:
        """Regular-session close (16:00 New York) as a UTC datetime."""
        d = self._require_session(d)
        return datetime.combine(d, REGULAR_SESSION_CLOSE, tzinfo=MARKET_TZ).astimezone(timezone.utc)

    def _require_session(self, d) -> date:
        d = _as_date(d)
        if not self.is_session(d):
            raise MarketDataError(f"{d} is not a trading session")
        return d
