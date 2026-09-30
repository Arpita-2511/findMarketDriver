"""
Leakage-safe daily sentiment features (Phase 5B, sentiment_features_v1).

Input : ScoredArticle = canonical NewsArticle + its FinBERT SentimentResult
Rows  : (symbol, trading_date) for completed market bars
Output: SentimentFeatureRow per row, in memory (no persistence)

Leakage rule (the only one), reused from services/news_alignment.py:

    article contributes to row (symbol, D)  iff
        symbol in article.symbols
        and is_eligible(article, prediction_timestamp(D))
            i.e. information_available_at <= prediction_timestamp(D)

    prediction_timestamp(D) = row_prediction_timestamp(D, calendar)
                            = completion time of bar D = D 16:30 America/New_York

created_at, updated_at, fetched_at, calendar dates and session labels are
never used for eligibility. There is no horizon input: the prediction
horizon only changes the target, never these features.

Semantics (v1): CUMULATIVE - every eligible article since the start of the
supplied news, as of prediction_timestamp. No rolling windows or decay yet.
Zero-news convention: all counts, ratios, means, std and maxima are 0.0
(never NaN) when news_count == 0.

This module performs no network calls, no model inference and no I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from datetime import date, datetime
from typing import Iterable

import pandas as pd

from services.finbert_sentiment import (
    LABELS,
    SentimentModelError,
    SentimentResult,
    build_sentiment_text,
    text_hash,
)
from services.market_calendar_service import TradingCalendar
from services.news_alignment import check_not_future, is_eligible, row_prediction_timestamp, to_exchange_time
from services.news_schema import NewsArticle

SENTIMENT_FEATURE_VERSION = "sentiment_features_v1"

FEATURE_COLUMNS = (
    "news_count",
    "positive_count",
    "negative_count",
    "neutral_count",
    "positive_ratio",
    "negative_ratio",
    "neutral_ratio",
    "mean_sentiment",
    "sentiment_std",
    "mean_positive_probability",
    "mean_negative_probability",
    "mean_neutral_probability",
    "max_positive_probability",
    "max_negative_probability",
    "max_neutral_probability",
)


class SentimentFeatureError(ValueError):
    """Invalid, inconsistent or duplicated input to sentiment feature generation."""


# ==========================================
# Input
# ==========================================


@dataclass(frozen=True)
class ScoredArticle:
    """A canonical article together with the FinBERT result for ITS text."""

    article: NewsArticle
    sentiment: SentimentResult

    def __post_init__(self):
        if not isinstance(self.article, NewsArticle):
            raise SentimentFeatureError("article must be a NewsArticle")
        if not isinstance(self.sentiment, SentimentResult):
            raise SentimentFeatureError(f"{self.article.article_key}: sentiment must be a SentimentResult")
        try:
            # Re-run the Phase 5A contract (range, sum = 1, label = argmax, score) -
            # guards against results altered after construction.
            SentimentResult(**self.sentiment.to_dict())
        except SentimentModelError as e:
            raise SentimentFeatureError(f"{self.article.article_key}: invalid sentiment result: {e}") from None
        if self.sentiment.input_text_hash != text_hash(build_sentiment_text(self.article)):
            raise SentimentFeatureError(
                f"{self.article.article_key}: sentiment was not computed from this article's text")
        check_not_future(self.article)

    @property
    def key(self) -> tuple[str, str]:
        return self.article.key


def _order(item: ScoredArticle):
    a = item.article
    return (a.information_available_at, a.provider, a.provider_article_id)


def prepare_scored_articles(items: Iterable[ScoredArticle]) -> list[ScoredArticle]:
    """
    Collapse identical duplicates of (provider, provider_article_id); raise on
    conflicting ones. Returns articles sorted by (information_available_at,
    provider, provider_article_id) - independent of input order.
    """
    unique: dict[tuple[str, str], ScoredArticle] = {}
    for item in items:
        if not isinstance(item, ScoredArticle):
            raise SentimentFeatureError("inputs must be ScoredArticle instances")
        existing = unique.get(item.key)
        if existing is None:
            unique[item.key] = item
        elif existing != item:
            raise SentimentFeatureError(
                f"conflicting records for article {item.article.article_key} "
                "(different article fields or sentiment); refusing to choose one")
    return sorted(unique.values(), key=_order)


# ==========================================
# Aggregation
# ==========================================


class _Accumulator:
    """Running sentiment_features_v1 statistics; add() must be called in a fixed order."""

    def __init__(self):
        self.n = 0
        self.label_counts = {label: 0 for label in LABELS}
        self.prob_sums = {label: 0.0 for label in LABELS}
        self.prob_max = {label: 0.0 for label in LABELS}
        self.score_mean = 0.0
        self.score_m2 = 0.0          # Welford: sum of squared deviations

    def add(self, sentiment: SentimentResult) -> None:
        self.n += 1
        self.label_counts[sentiment.label] += 1
        for label, p in sentiment.probabilities().items():
            self.prob_sums[label] += p
            self.prob_max[label] = max(self.prob_max[label], p)
        delta = sentiment.sentiment_score - self.score_mean
        self.score_mean += delta / self.n
        self.score_m2 += delta * (sentiment.sentiment_score - self.score_mean)

    def features(self) -> dict[str, float]:
        n = self.n
        if n == 0:                   # zero-news convention: everything 0, never NaN
            return {name: (0 if name.endswith("_count") else 0.0) for name in FEATURE_COLUMNS}
        return {
            "news_count": n,
            "positive_count": self.label_counts["positive"],
            "negative_count": self.label_counts["negative"],
            "neutral_count": self.label_counts["neutral"],
            "positive_ratio": self.label_counts["positive"] / n,
            "negative_ratio": self.label_counts["negative"] / n,
            "neutral_ratio": self.label_counts["neutral"] / n,
            "mean_sentiment": self.score_mean,
            "sentiment_std": math.sqrt(max(self.score_m2 / n, 0.0)),     # population std
            "mean_positive_probability": self.prob_sums["positive"] / n,
            "mean_negative_probability": self.prob_sums["negative"] / n,
            "mean_neutral_probability": self.prob_sums["neutral"] / n,
            "max_positive_probability": self.prob_max["positive"],
            "max_negative_probability": self.prob_max["negative"],
            "max_neutral_probability": self.prob_max["neutral"],
        }


# ==========================================
# Output
# ==========================================


@dataclass(frozen=True)
class SentimentFeatureRow:
    symbol: str
    trading_date: date
    prediction_timestamp: datetime        # timezone-aware UTC
    sentiment_feature_version: str
    news_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    mean_sentiment: float
    sentiment_std: float
    mean_positive_probability: float
    mean_negative_probability: float
    mean_neutral_probability: float
    max_positive_probability: float
    max_negative_probability: float
    max_neutral_probability: float

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


OUTPUT_COLUMNS = tuple(f.name for f in fields(SentimentFeatureRow))


def _normalize_symbols(symbols: Iterable[str]) -> list[str]:
    if isinstance(symbols, str):
        raise SentimentFeatureError("symbols must be a sequence of tickers, not a string")
    out = sorted({str(s).strip().upper() for s in symbols})
    if not out or not all(out):
        raise SentimentFeatureError("at least one non-empty symbol is required")
    return out


def _row_time(trading_date, calendar: TradingCalendar) -> tuple[datetime, date]:
    """
    (prediction_timestamp UTC, trading session date) for a completed-bar date.
    Validation (session only, no time-of-day / timezone values) is done by
    row_prediction_timestamp / TradingCalendar; the session date is read back
    from the timestamp on the New York clock.
    """
    timestamp = row_prediction_timestamp(trading_date, calendar)
    return timestamp, to_exchange_time(timestamp).date()


def _row(symbol, trading_date, prediction_timestamp, acc: _Accumulator) -> SentimentFeatureRow:
    return SentimentFeatureRow(symbol=symbol, trading_date=trading_date, prediction_timestamp=prediction_timestamp,
                               sentiment_feature_version=SENTIMENT_FEATURE_VERSION, **acc.features())


# ==========================================
# Public API
# ==========================================


def generate_sentiment_features(
    scored_articles: Iterable[ScoredArticle],
    trading_dates: Iterable,
    calendar: TradingCalendar,
    symbols: Iterable[str],
) -> list[SentimentFeatureRow]:
    """
    Cumulative sentiment_features_v1 for every (symbol, trading_date).

    trading_dates must be sessions of `calendar` (completed-bar dates);
    weekends/holidays raise instead of creating artificial rows. Output is
    sorted by (symbol, trading_date) and does not depend on input order.

    Chronological single pass per symbol: articles sorted by availability,
    rows by prediction timestamp; a pointer admits each article exactly once,
    when is_eligible(article, row timestamp) first becomes true.
    """
    articles = prepare_scored_articles(scored_articles)
    wanted = _normalize_symbols(symbols)

    rows_in_time = sorted(_row_time(d, calendar) for d in trading_dates)
    if len({d for _, d in rows_in_time}) != len(rows_in_time):
        raise SentimentFeatureError("trading_dates contains duplicates")

    output: list[SentimentFeatureRow] = []
    for symbol in wanted:
        relevant = [a for a in articles if symbol in a.article.symbols]
        acc, i = _Accumulator(), 0
        for timestamp, trading_date in rows_in_time:
            while i < len(relevant) and is_eligible(relevant[i].article, timestamp):
                acc.add(relevant[i].sentiment)
                i += 1
            output.append(_row(symbol, trading_date, timestamp, acc))
    return output


def features_as_of(scored_articles: Iterable[ScoredArticle], trading_date, calendar: TradingCalendar,
                   symbol: str) -> SentimentFeatureRow:
    """Direct (non-incremental) computation for one row - same definitions, same order."""
    [symbol] = _normalize_symbols([symbol])
    timestamp, session = _row_time(trading_date, calendar)
    acc = _Accumulator()
    for item in prepare_scored_articles(scored_articles):
        if symbol in item.article.symbols and is_eligible(item.article, timestamp):
            acc.add(item.sentiment)
    return _row(symbol, session, timestamp, acc)


def features_to_frame(rows: Iterable[SentimentFeatureRow]) -> pd.DataFrame:
    """DataFrame with OUTPUT_COLUMNS; prediction_timestamp stays timezone-aware (UTC)."""
    frame = pd.DataFrame([r.to_dict() for r in rows], columns=list(OUTPUT_COLUMNS))
    frame["prediction_timestamp"] = pd.to_datetime(frame["prediction_timestamp"], utc=True)
    return frame
