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
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from services.market_data import EXCHANGE_TIMEZONE, MarketDataError

MARKET_TZ = ZoneInfo(EXCHANGE_TIMEZONE)
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
