"""
Leakage-safe sentiment-window features (Phase 9, sentiment_window_v1).

A read-only representation derived from the VERIFIED Phase 5C/8 sentiment
records (services/historical_sentiment.load_sentiment_dataset). Nothing is
rescored and no existing dataset is modified.

Why: sentiment_features_v1 is cumulative since the start of the news, so its
counts grow without bound and every walk-forward test fold lies outside the
training range (ARCHITECTURE.md 15.2). These features use bounded,
session-based windows instead - the same window semantics as the Phase 6
event features:

    ts(D)        = row_prediction_timestamp(D) = D 16:30 America/New_York
    prev_k(D)    = k-th trading session before D (TradingCalendar)
    W_k(D)       = { article : symbol in article.symbols and
                                ts(prev_k(D)) < information_available_at <= ts(D) }

The upper bound is the Phase 4B eligibility rule (is_eligible); the lower
bound only removes older articles. W_1 is exactly the Phase 6 per-session
window. Windows: k = 1, 5, 20 sessions.

Features (fixed before any Phase 9 result was observed):

    sw_count_k             |W_k|                                   k = 1, 5, 20
    sw_mean_k              mean FinBERT sentiment_score over W_k   k = 1, 5, 20
    sw_positive_ratio_k    share of W_k labelled positive          k = 5, 20
    sw_negative_ratio_k    share of W_k labelled negative          k = 5, 20
    sw_mean_change_5_20    sw_mean_5 - sw_mean_20
    sw_count_surprise_1_20 sw_count_1 - sw_count_20 / 20   (news volume vs its 20-session average)
    sw_mean_surprise_1_20  sw_mean_1 - sw_mean_20          (today's tone vs the 20-session tone)

sentiment_score = P(positive) - P(negative) (Phase 5A). Empty-window
convention (as sentiment_features_v1 / event_features_v1): means and ratios
are 0.0 when the window is empty; the count says it was empty.

Coverage: a row is covered only if its LONGEST window lies inside the news
dataset interval, i.e. ts(prev_20(D)) >= interval start and ts(D) <= interval
end. Uncovered rows get NaN (unknown) and covered = False - a partially
observed window is never reported as "little news". Known edge effect: an
article created before the interval start but revised into it is not in the
dataset (selection is by created_at), as for every Phase 5B/6 feature.
"""

from __future__ import annotations

import bisect
import math
from datetime import datetime
from typing import Iterable

import pandas as pd

from services.market_calendar_service import TradingCalendar
from services.news_alignment import is_eligible, require_aware, row_prediction_timestamp, to_exchange_time
from services.sentiment_features import ScoredArticle, prepare_scored_articles

SENTIMENT_WINDOW_VERSION = "sentiment_window_v1"
WINDOWS = (1, 5, 20)
LONGEST_WINDOW = max(WINDOWS)

COLUMNS = (
    "sw_count_1", "sw_count_5", "sw_count_20",
    "sw_mean_1", "sw_mean_5", "sw_mean_20",
    "sw_positive_ratio_5", "sw_negative_ratio_5",
    "sw_positive_ratio_20", "sw_negative_ratio_20",
    "sw_mean_change_5_20",
    "sw_count_surprise_1_20",
    "sw_mean_surprise_1_20",
)

DEFINITIONS = {
    "window": ("W_k(D) = articles with symbol in symbols and "
               "ts(prev_k(D)) < information_available_at <= ts(D); ts(D) = D 16:30 America/New_York; "
               "prev_k(D) = k-th trading session before D"),
    "sw_count_k": "|W_k(D)|, k in (1, 5, 20)",
    "sw_mean_k": "mean of FinBERT sentiment_score (P(pos) - P(neg)) over W_k(D); 0.0 if empty; k in (1, 5, 20)",
    "sw_positive_ratio_k": "#positive-label articles in W_k / |W_k|; 0.0 if empty; k in (5, 20)",
    "sw_negative_ratio_k": "#negative-label articles in W_k / |W_k|; 0.0 if empty; k in (5, 20)",
    "sw_mean_change_5_20": "sw_mean_5 - sw_mean_20",
    "sw_count_surprise_1_20": "sw_count_1 - sw_count_20 / 20",
    "sw_mean_surprise_1_20": "sw_mean_1 - sw_mean_20",
    "coverage": ("covered iff ts(prev_20(D)) >= news interval start and ts(D) <= news interval end; "
                 "uncovered rows are NaN"),
    "information_timestamp": "every input article has information_available_at <= ts(D)",
    "source": "verified sentiment_records_v1 (Phase 5C/8), read-only",
}


class SentimentWindowError(ValueError):
    """Invalid input to the sentiment-window features."""


def _window_stats(items: list[ScoredArticle]) -> tuple[int, float, float, float]:
    n = len(items)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    mean = math.fsum(s.sentiment.sentiment_score for s in items) / n
    pos = sum(1 for s in items if s.sentiment.label == "positive") / n
    neg = sum(1 for s in items if s.sentiment.label == "negative") / n
    return n, mean, pos, neg


def generate_sentiment_window_features(
    scored_articles: Iterable[ScoredArticle],
    trading_dates: Iterable,
    calendar: TradingCalendar,
    symbol: str,
    *,
    coverage_start: datetime,
    coverage_end: datetime,
) -> pd.DataFrame:
    """
    One row per trading date: trading_date, prediction_timestamp (UTC),
    covered, COLUMNS. Deterministic and independent of input order.
    """
    coverage_start = require_aware(coverage_start, "coverage_start")
    coverage_end = require_aware(coverage_end, "coverage_end")
    symbol = str(symbol).strip().upper()
    if not symbol:
        raise SentimentWindowError("symbol is required")

    relevant = [s for s in prepare_scored_articles(scored_articles) if symbol in s.article.symbols]
    times = [s.article.information_available_at for s in relevant]

    def window(lower, upper) -> list[ScoredArticle]:
        lo, hi = bisect.bisect_right(times, lower), bisect.bisect_right(times, upper)
        items = relevant[lo:hi]
        if not all(is_eligible(s.article, upper) for s in items):        # Phase 4B rule, defensive
            raise SentimentWindowError("internal error: window contains an article not eligible at ts(D)")
        return items

    dates = list(trading_dates)
    if len(set(dates)) != len(dates):
        raise SentimentWindowError("trading_dates contains duplicates")

    rows = []
    for d in sorted(dates):
        ts = row_prediction_timestamp(d, calendar)                     # raises for non-sessions
        session = to_exchange_time(ts).date()
        prev = {k: calendar.previous_session_before(session, k) for k in WINDOWS}
        lower = {k: row_prediction_timestamp(p, calendar) if p is not None else None for k, p in prev.items()}
        covered = (lower[LONGEST_WINDOW] is not None and lower[LONGEST_WINDOW] >= coverage_start
                   and ts <= coverage_end)
        row = {"trading_date": session, "prediction_timestamp": ts, "covered": covered}
        if not covered:
            row.update({c: math.nan for c in COLUMNS})
            rows.append(row)
            continue
        stats = {k: _window_stats(window(lower[k], ts)) for k in WINDOWS}
        for k in WINDOWS:
            row[f"sw_count_{k}"] = stats[k][0]
            row[f"sw_mean_{k}"] = stats[k][1]
        for k in (5, 20):
            row[f"sw_positive_ratio_{k}"] = stats[k][2]
            row[f"sw_negative_ratio_{k}"] = stats[k][3]
        row["sw_mean_change_5_20"] = row["sw_mean_5"] - row["sw_mean_20"]
        row["sw_count_surprise_1_20"] = row["sw_count_1"] - row["sw_count_20"] / 20
        row["sw_mean_surprise_1_20"] = row["sw_mean_1"] - row["sw_mean_20"]
        rows.append(row)

    frame = pd.DataFrame(rows, columns=["trading_date", "prediction_timestamp", "covered", *COLUMNS])
    for c in COLUMNS:
        frame[c] = frame[c].astype("float64")
    return frame
