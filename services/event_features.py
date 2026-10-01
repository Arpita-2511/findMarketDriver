"""
Leakage-safe daily event features (Phase 6, event_features_v1). Pure, in memory.

Input : ClassifiedArticle = ScoredArticle (Phase 5B-validated article + FinBERT
        sentiment) + EventPrediction (services/event_classifier.py)
Rows  : (symbol, completed-bar trading date D)

Window semantics (differs deliberately from the cumulative sentiment_features_v1):

    ts(D)       = row_prediction_timestamp(D) = D 16:30 America/New_York
    prev(D)     = ts(previous calendar session), or none before the first session
    row window  : prev(D) < information_available_at <= ts(D)
    recent      : ts(5th previous session) < information_available_at <= ts(D)

Each article therefore falls into exactly ONE row - the first row whose
prediction timestamp is at/after its availability. Upper bound =
the Phase 4B eligibility rule (is_eligible); window bounds come from the
TradingCalendar, never from which dates were requested. Counts are per
session, so they do not grow with elapsed time.

"Events" = articles whose event_type is not OTHER. Impact comes from the
article's FinBERT label (POSITIVE/NEGATIVE/NEUTRAL), event_impact_score is the
mean FinBERT sentiment_score of the window's events.

Zero convention: every count/score is 0 (never NaN); dominant_event_type is
"NONE" when the window has no events.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Iterable

from services.event_classifier import EventPrediction
from services.event_taxonomy import EVENT_TYPES, NO_EVENT, impact_from_sentiment_label, type_feature_name
from services.market_calendar_service import TradingCalendar
from services.news_alignment import is_eligible, row_prediction_timestamp, to_exchange_time
from services.news_schema import NewsArticle
from services.sentiment_features import ScoredArticle

EVENT_FEATURE_VERSION = "event_features_v1"
RECENT_SESSIONS = 5
NO_DOMINANT_EVENT = "NONE"

EVENT_TYPE_COLUMNS = tuple(type_feature_name(t) for t in EVENT_TYPES if t != NO_EVENT)

EVENT_FEATURE_COLUMNS = (
    "article_count",
    "event_count",
    "unique_event_type_count",
    *EVENT_TYPE_COLUMNS,
    "positive_event_count",
    "negative_event_count",
    "neutral_event_count",
    "event_impact_score",
    "mean_event_confidence",
    "dominant_event_type",
    "recent_event_count",
)
EVENT_OUTPUT_COLUMNS = ("symbol", "trading_date", "prediction_timestamp", "event_feature_version",
                        *EVENT_FEATURE_COLUMNS)


class EventFeatureError(ValueError):
    """Invalid or conflicting input to event feature generation."""


@dataclass(frozen=True)
class ClassifiedArticle:
    scored: ScoredArticle              # already validated by Phase 5B (sentiment + text hash + not future)
    prediction: EventPrediction

    def __post_init__(self):
        if not isinstance(self.scored, ScoredArticle) or not isinstance(self.prediction, EventPrediction):
            raise EventFeatureError("ClassifiedArticle needs a ScoredArticle and an EventPrediction")

    @property
    def article(self) -> NewsArticle:
        return self.scored.article

    @property
    def key(self) -> tuple[str, str]:
        return self.scored.article.key

    @property
    def event_type(self) -> str:
        return self.prediction.event_type

    @property
    def event_confidence(self) -> float:
        return self.prediction.event_confidence

    @property
    def event_impact(self) -> str:
        return impact_from_sentiment_label(self.scored.sentiment.label)

    @property
    def event_impact_score(self) -> float:
        return self.scored.sentiment.sentiment_score

    @property
    def is_event(self) -> bool:
        return self.prediction.event_type != NO_EVENT


def _order(c: ClassifiedArticle):
    a = c.article
    return (a.information_available_at, a.provider, a.provider_article_id)


def prepare_classified(items: Iterable[ClassifiedArticle]) -> list[ClassifiedArticle]:
    """Identical duplicates collapse, conflicting duplicates raise; sorted by availability."""
    unique: dict[tuple[str, str], ClassifiedArticle] = {}
    for item in items:
        if not isinstance(item, ClassifiedArticle):
            raise EventFeatureError("inputs must be ClassifiedArticle instances")
        existing = unique.get(item.key)
        if existing is None:
            unique[item.key] = item
        elif existing != item:
            raise EventFeatureError(f"conflicting records for {item.article.article_key}")
    return sorted(unique.values(), key=_order)


def _window_features(window: list[ClassifiedArticle]) -> dict:
    events = [c for c in window if c.is_event]
    type_counts = {t: 0 for t in EVENT_TYPES if t != NO_EVENT}
    for c in events:
        type_counts[c.event_type] += 1
    n = len(events)
    dominant = NO_DOMINANT_EVENT
    if n:
        dominant = max(type_counts, key=lambda t: (type_counts[t], -EVENT_TYPES.index(t)))
    return {
        "article_count": len(window),
        "event_count": n,
        "unique_event_type_count": sum(1 for v in type_counts.values() if v),
        **{type_feature_name(t): v for t, v in type_counts.items()},
        "positive_event_count": sum(1 for c in events if c.event_impact == "POSITIVE"),
        "negative_event_count": sum(1 for c in events if c.event_impact == "NEGATIVE"),
        "neutral_event_count": sum(1 for c in events if c.event_impact == "NEUTRAL"),
        "event_impact_score": sum(c.event_impact_score for c in events) / n if n else 0.0,
        "mean_event_confidence": sum(c.event_confidence for c in events) / n if n else 0.0,
        "dominant_event_type": dominant,
    }


def _session_ts(calendar: TradingCalendar, d, n: int):
    prev = calendar.previous_session_before(d, n)
    return row_prediction_timestamp(prev, calendar) if prev is not None else None


def _slice(items, times, lower, upper):
    lo = bisect.bisect_right(times, lower) if lower is not None else 0
    hi = bisect.bisect_right(times, upper)
    window = items[lo:hi]
    # every member must satisfy the Phase 4B rule; the window only adds a lower bound
    if not all(is_eligible(c.article, upper) for c in window):
        raise EventFeatureError("internal error: window contains an article not eligible at its prediction time")
    return window


def generate_event_features(items: Iterable[ClassifiedArticle], trading_dates: Iterable,
                            calendar: TradingCalendar, symbols: Iterable[str]) -> list[dict]:
    """event_features_v1 rows (dicts in EVENT_OUTPUT_COLUMNS order), sorted by (symbol, trading_date)."""
    classified = prepare_classified(items)
    if isinstance(symbols, str):
        raise EventFeatureError("symbols must be a sequence of tickers, not a string")
    wanted = sorted({str(s).strip().upper() for s in symbols})
    if not wanted or not all(wanted):
        raise EventFeatureError("at least one non-empty symbol is required")

    rows_in_time = []
    for d in trading_dates:
        ts = row_prediction_timestamp(d, calendar)            # validates: completed-bar session only
        session = to_exchange_time(ts).date()
        rows_in_time.append((ts, session))
    rows_in_time.sort()
    if len({s for _, s in rows_in_time}) != len(rows_in_time):
        raise EventFeatureError("trading_dates contains duplicates")

    output = []
    for symbol in wanted:
        relevant = [c for c in classified if symbol in c.article.symbols]
        times = [c.article.information_available_at for c in relevant]
        for ts, session in rows_in_time:
            window = _slice(relevant, times, _session_ts(calendar, session, 1), ts)
            recent = _slice(relevant, times, _session_ts(calendar, session, RECENT_SESSIONS), ts)
            row = {"symbol": symbol, "trading_date": session, "prediction_timestamp": ts,
                   "event_feature_version": EVENT_FEATURE_VERSION, **_window_features(window),
                   "recent_event_count": sum(1 for c in recent if c.is_event)}
            output.append({k: row[k] for k in EVENT_OUTPUT_COLUMNS})
    return output
