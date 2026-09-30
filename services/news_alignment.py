"""
Temporal alignment of canonical news with trading sessions and
prediction timestamps (Phase 4B). Provider-independent: uses only
NewsArticle fields (services/news_schema.py).

THE eligibility rule (the only one):

    an article may inform a prediction made at prediction_timestamp
    if and only if  information_available_at <= prediction_timestamp

Session assignment (assign_session) is a descriptive label - which
trading session an article falls into - and is NEVER used to decide
eligibility: news at 10:30 belongs to that day's session but must not
reach a 09:30 prediction of the same day.

Two separate concepts (ARCHITECTURE.md section 12.1):

    news availability    information_available_at of the article
    daily-bar completion D 16:30 New York (market_calendar_service);
                         only defines WHEN a daily row's prediction is
                         made, never when news became available

direction_v1 integration: a row t uses the completed bar of day t, so its
prediction timestamp is row_prediction_timestamp(t) = completion_time(t).
The horizon h only moves the label's close (Close[t+h]); it never moves
the prediction timestamp, so news arriving in (t, t+h] is never eligible
for row t.

All datetimes are timezone-aware; they are kept in UTC and converted to
America/New_York only to interpret calendar dates and session phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Iterable

from services.market_calendar_service import (
    MARKET_TZ,
    REGULAR_SESSION_CLOSE,
    REGULAR_SESSION_OPEN,
    TradingCalendar,
    completion_time,
)
from services.market_data import MarketDataError
from services.news_schema import NewsArticle


class FutureInformationError(ValueError):
    """Information that cannot have existed at the time it is claimed to be used."""


class SessionPhase(str, Enum):
    PRE_MARKET = "pre_market"            # trading day, before 09:30 New York
    REGULAR = "regular"                  # trading day, 09:30 .. 16:00 inclusive
    POST_MARKET = "post_market"          # trading day, after 16:00
    NON_TRADING_DAY = "non_trading_day"  # weekend / holiday


# ==========================================
# Time helpers
# ==========================================


def require_aware(value: datetime, name: str) -> datetime:
    """Reject naive datetimes; return the value in UTC."""
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def to_exchange_time(value: datetime) -> datetime:
    """Exchange-local (America/New_York) view of an aware timestamp; zoneinfo applies EST/EDT."""
    return require_aware(value, "timestamp").astimezone(MARKET_TZ)


def check_not_future(article: NewsArticle) -> None:
    """
    A stored version cannot become available after we retrieved it.
    information_available_at > fetched_at means corrupt or clock-skewed data.
    """
    if article.information_available_at > article.fetched_at:
        raise FutureInformationError(
            f"{article.article_key}: information_available_at {article.information_available_at.isoformat()} "
            f"is after fetched_at {article.fetched_at.isoformat()}"
        )


# ==========================================
# Eligibility
# ==========================================


def is_eligible(article: NewsArticle, prediction_timestamp: datetime) -> bool:
    """information_available_at <= prediction_timestamp (created_at/updated_at are not used)."""
    check_not_future(article)
    return article.information_available_at <= require_aware(prediction_timestamp, "prediction_timestamp")


def _order(article: NewsArticle):
    return (article.information_available_at, article.provider, article.provider_article_id)


def eligible_articles(
    articles: Iterable[NewsArticle],
    prediction_timestamp: datetime,
    *,
    symbol: str | None = None,
    not_before: datetime | None = None,
    now: datetime | None = None,
) -> list[NewsArticle]:
    """
    Articles usable by a prediction made at `prediction_timestamp`,
    ordered by (information_available_at, provider, provider_article_id).

    symbol     keep only articles whose canonical `symbols` contain it
               (a multi-symbol article is available to each listed symbol;
               symbols are never inferred)
    not_before optional EXCLUSIVE lower bound on information_available_at,
               for lookback windows: not_before < available <= prediction
    now        if given (live use), a prediction_timestamp later than `now`
               is rejected - it would consume news that does not exist yet
    """
    prediction_timestamp = require_aware(prediction_timestamp, "prediction_timestamp")
    if now is not None and prediction_timestamp > require_aware(now, "now"):
        raise FutureInformationError(
            f"prediction_timestamp {prediction_timestamp.isoformat()} is in the future"
        )
    lower = require_aware(not_before, "not_before") if not_before is not None else None
    if lower is not None and lower >= prediction_timestamp:
        raise ValueError("not_before must be earlier than prediction_timestamp")
    wanted = symbol.strip().upper() if symbol is not None else None

    selected = []
    for article in articles:
        check_not_future(article)
        if article.information_available_at > prediction_timestamp:
            continue
        if lower is not None and article.information_available_at <= lower:
            continue
        if wanted is not None and wanted not in article.symbols:
            continue
        selected.append(article)
    return sorted(selected, key=_order)


# ==========================================
# Session assignment (descriptive, not eligibility)
# ==========================================


@dataclass(frozen=True)
class SessionAssignment:
    article_key: str
    symbols: tuple[str, ...]
    created_at: datetime                   # unchanged from the article
    information_available_at: datetime     # UTC - the timestamp that was assigned
    exchange_local_time: datetime          # same instant, America/New_York
    calendar_date: date                    # exchange-local date of availability
    phase: SessionPhase
    session_date: date                     # trading session the article falls into


def classify_phase(local_time: datetime, calendar: TradingCalendar) -> SessionPhase:
    """
    Phase of an instant; regular session is 09:30..16:00 inclusive, read
    on the New York clock. Any aware timestamp is converted first; naive
    ones are rejected (never interpreted in a machine-local timezone).
    """
    local_time = to_exchange_time(local_time)
    d = local_time.date()
    if not calendar.is_session(d):
        return SessionPhase.NON_TRADING_DAY
    t = local_time.timetz().replace(tzinfo=None)
    if t < REGULAR_SESSION_OPEN:
        return SessionPhase.PRE_MARKET
    if t <= REGULAR_SESSION_CLOSE:
        return SessionPhase.REGULAR
    return SessionPhase.POST_MARKET


def assign_session(article: NewsArticle, calendar: TradingCalendar) -> SessionAssignment:
    """
    Trading session an article falls into, based on information_available_at
    (REQ-NEWS-005, cutoff = 16:00 regular close):

        pre-market / regular (<= 16:00) on a trading day D  -> D
        post-market (> 16:00) on a trading day D            -> next session after D
        weekend / holiday                                    -> next session after that date
    """
    check_not_future(article)
    local = to_exchange_time(article.information_available_at)
    phase = classify_phase(local, calendar)
    if phase in (SessionPhase.PRE_MARKET, SessionPhase.REGULAR):
        session = local.date()
    else:
        session = calendar.next_session_after(local.date())
    return SessionAssignment(
        article_key=article.article_key,
        symbols=article.symbols,
        created_at=article.created_at,
        information_available_at=article.information_available_at,
        exchange_local_time=local,
        calendar_date=local.date(),
        phase=phase,
        session_date=session,
    )


def assign_sessions(articles: Iterable[NewsArticle], calendar: TradingCalendar) -> list[SessionAssignment]:
    """Deterministic: ordered by (information_available_at, provider, provider_article_id)."""
    return [assign_session(a, calendar) for a in sorted(articles, key=_order)]


# ==========================================
# direction_v1 row integration
# ==========================================


def row_prediction_timestamp(bar_date, calendar: TradingCalendar) -> datetime:
    """
    Prediction timestamp (UTC) of the feature row for trading session
    `bar_date`: the moment its daily bar is complete (16:00 New York +
    settlement delay; market_calendar_service.completion_time). News
    eligible for that row must have information_available_at <= this.
    Independent of the prediction horizon.
    """
    if not calendar.is_session(bar_date):
        raise MarketDataError(f"{bar_date} is not a trading session")
    return completion_time(bar_date).astimezone(timezone.utc)


def row_prediction_timestamps(bar_dates: Iterable, calendar: TradingCalendar) -> list[datetime]:
    return [row_prediction_timestamp(d, calendar) for d in bar_dates]
