"""
Phase 5B: leakage-safe daily sentiment features (sentiment_features_v1).
Offline: hand-built SentimentResults, no model, no network.
"""

import math
import random
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from services.finbert_sentiment import INFERENCE_VERSION, LABELS, SentimentResult, build_sentiment_text, text_hash
from services.market_calendar_service import TradingCalendar
from services.market_data import MarketDataError
from services.news_schema import NewsArticle, historical_information_available_at
from services.sentiment_features import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    SENTIMENT_FEATURE_VERSION,
    ScoredArticle,
    SentimentFeatureError,
    features_as_of,
    features_to_frame,
    generate_sentiment_features,
)
from tests.conftest import calendar_2024
from training.evaluation_harness import build_targets

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
CAL = calendar_2024()
FETCHED = datetime(2026, 10, 1, tzinfo=UTC)

POS = (0.7, 0.1, 0.2)     # (positive, negative, neutral)
NEG = (0.1, 0.8, 0.1)
NEU = (0.2, 0.1, 0.7)


def ny(*args):
    return datetime(*args, tzinfo=NY)


def scored(i, when, probs=POS, symbols=("AAPL",), updated=None, headline=None):
    article = NewsArticle(
        provider="alpaca", provider_article_id=str(i), headline=headline or f"Headline {i}",
        symbols=symbols, created_at=when, updated_at=updated,
        information_available_at=historical_information_available_at(when, updated), fetched_at=FETCHED)
    pos, neg, neu = probs
    label = max(LABELS, key=lambda lbl: ({"positive": pos, "negative": neg, "neutral": neu}[lbl], -LABELS.index(lbl)))
    sentiment = SentimentResult(
        inference_version=INFERENCE_VERSION, model_name="ProsusAI/finbert", model_revision="rev",
        label=label, positive_probability=pos, negative_probability=neg, neutral_probability=neu,
        sentiment_score=pos - neg, input_text_hash=text_hash(build_sentiment_text(article)))
    return ScoredArticle(article, sentiment)


def one(rows, symbol="AAPL", day=None):
    [row] = [r for r in rows if r.symbol == symbol and (day is None or r.trading_date == day)]
    return row


def gen(items, dates, symbols=("AAPL",), calendar=CAL):
    return generate_sentiment_features(items, dates, calendar, symbols)


D = date(2024, 7, 10)                       # Wednesday; row timestamp 16:30 EDT = 20:30 UTC


# ==========================================
# Feature values
# ==========================================


def test_zero_news_row_is_all_zero_not_nan():
    row = one(gen([], [D]))
    for name in FEATURE_COLUMNS:
        value = getattr(row, name)
        assert value == 0 and not (isinstance(value, float) and math.isnan(value))
    assert isinstance(row.news_count, int)


def test_one_article():
    row = one(gen([scored(1, ny(2024, 7, 10, 10, 0), POS)], [D]))
    assert (row.news_count, row.positive_count, row.negative_count, row.neutral_count) == (1, 1, 0, 0)
    assert (row.positive_ratio, row.negative_ratio, row.neutral_ratio) == (1.0, 0.0, 0.0)
    assert row.mean_sentiment == pytest.approx(0.6) and row.sentiment_std == 0.0
    assert (row.max_positive_probability, row.max_negative_probability, row.max_neutral_probability) == POS


def test_multiple_articles_counts_ratios_means_std_maxima():
    items = [scored(1, ny(2024, 7, 10, 9, 0), POS), scored(2, ny(2024, 7, 10, 10, 0), NEG),
             scored(3, ny(2024, 7, 10, 11, 0), NEU), scored(4, ny(2024, 7, 10, 12, 0), POS)]
    row = one(gen(items, [D]))
    scores = [0.6, -0.7, 0.1, 0.6]
    assert (row.news_count, row.positive_count, row.negative_count, row.neutral_count) == (4, 2, 1, 1)
    assert (row.positive_ratio, row.negative_ratio, row.neutral_ratio) == (0.5, 0.25, 0.25)
    assert row.mean_sentiment == pytest.approx(np.mean(scores))
    assert row.sentiment_std == pytest.approx(np.std(scores))                  # population std
    assert row.mean_positive_probability == pytest.approx((0.7 + 0.1 + 0.2 + 0.7) / 4)
    assert row.mean_negative_probability == pytest.approx((0.1 + 0.8 + 0.1 + 0.1) / 4)
    assert row.mean_neutral_probability == pytest.approx((0.2 + 0.1 + 0.7 + 0.2) / 4)
    assert (row.max_positive_probability, row.max_negative_probability, row.max_neutral_probability) == (0.7, 0.8, 0.7)


# ==========================================
# Duplicates & validation
# ==========================================


def test_identical_duplicates_collapse():
    a = scored(1, ny(2024, 7, 10, 10, 0))
    assert one(gen([a, a, scored(1, ny(2024, 7, 10, 10, 0))], [D])).news_count == 1


def test_conflicting_duplicates_raise():
    base = scored(1, ny(2024, 7, 10, 10, 0), POS)
    with pytest.raises(SentimentFeatureError, match="conflicting"):
        gen([base, scored(1, ny(2024, 7, 10, 10, 0), NEG)], [D])                     # different sentiment
    with pytest.raises(SentimentFeatureError, match="conflicting"):
        gen([base, scored(1, ny(2024, 7, 10, 10, 0), POS, updated=ny(2024, 7, 10, 11, 0))], [D])


def test_different_ids_with_same_headline_are_not_merged():
    items = [scored(1, ny(2024, 7, 10, 10, 0), headline="Same"), scored(2, ny(2024, 7, 10, 10, 0), headline="Same")]
    assert one(gen(items, [D])).news_count == 2


def test_label_inconsistent_with_probabilities_rejected():
    item = scored(1, ny(2024, 7, 10, 10, 0), POS)
    object.__setattr__(item.sentiment, "label", "negative")                    # tamper after construction
    with pytest.raises(SentimentFeatureError, match="invalid sentiment"):
        ScoredArticle(item.article, item.sentiment)


def test_sentiment_from_another_article_rejected():
    a, b = scored(1, ny(2024, 7, 10, 10, 0)), scored(2, ny(2024, 7, 10, 10, 0), headline="Other text")
    with pytest.raises(SentimentFeatureError, match="not computed from this article"):
        ScoredArticle(a.article, b.sentiment)


def test_wrong_input_types_rejected():
    with pytest.raises(SentimentFeatureError):
        gen([("not", "scored")], [D])
    with pytest.raises(SentimentFeatureError):
        ScoredArticle(scored(1, ny(2024, 7, 10, 10, 0)).article, {"label": "positive"})


# ==========================================
# Symbols
# ==========================================


def test_symbol_filtering_and_multi_symbol_articles():
    items = [scored(1, ny(2024, 7, 10, 9, 0), symbols=("AAPL",)),
             scored(2, ny(2024, 7, 10, 10, 0), NEG, symbols=("GOOG",)),
             scored(3, ny(2024, 7, 10, 11, 0), NEU, symbols=("AAPL", "MSFT"))]
    rows = gen(items, [D], symbols=["msft", "AAPL"])
    assert [r.symbol for r in rows] == ["AAPL", "MSFT"]                           # sorted, normalized
    assert one(rows, "AAPL").news_count == 2                                     # 1 and 3; GOOG excluded
    assert one(rows, "MSFT").news_count == 1 and one(rows, "MSFT").neutral_count == 1
    assert one(gen(items, [D], symbols=["TSLA"]), "TSLA").news_count == 0


def test_invalid_symbol_requests():
    with pytest.raises(SentimentFeatureError):
        gen([], [D], symbols="AAPL")
    with pytest.raises(SentimentFeatureError):
        gen([], [D], symbols=[])


# ==========================================
# Timing: boundary, post-market, weekend, holiday, DST
# ==========================================


def test_boundary_equality_is_eligible_one_second_later_is_not():
    at = scored(1, ny(2024, 7, 10, 16, 30))
    after = scored(2, ny(2024, 7, 10, 16, 30, 1))
    row = one(gen([at, after], [D]))
    assert row.prediction_timestamp == datetime(2024, 7, 10, 20, 30, tzinfo=UTC)
    assert row.news_count == 1


def test_post_market_example_from_the_specification():
    cal = TradingCalendar(d.date() for d in pd.bdate_range("2026-08-31", "2026-09-30")
                          if d.date() != date(2026, 9, 7))                        # Labor Day
    a = scored(1, ny(2026, 9, 10, 15, 0))
    b = scored(2, ny(2026, 9, 10, 17, 0))
    rows = gen([a, b], [date(2026, 9, 10), date(2026, 9, 11)], calendar=cal)
    assert one(rows, day=date(2026, 9, 10)).news_count == 1                      # only A
    assert one(rows, day=date(2026, 9, 11)).news_count == 2                      # B joins the next row


def test_weekend_article_joins_first_later_row():
    weekend = scored(1, ny(2024, 7, 13, 11, 0))                                  # Saturday
    rows = gen([weekend], [date(2024, 7, 12), date(2024, 7, 15)])
    assert one(rows, day=date(2024, 7, 12)).news_count == 0
    assert one(rows, day=date(2024, 7, 15)).news_count == 1


def test_holiday_article_and_holiday_rows():
    holiday = scored(1, ny(2024, 7, 4, 10, 0))                                   # Independence Day
    rows = gen([holiday], [date(2024, 7, 3), date(2024, 7, 5)])
    assert one(rows, day=date(2024, 7, 3)).news_count == 0
    assert one(rows, day=date(2024, 7, 5)).news_count == 1
    with pytest.raises(MarketDataError, match="not a trading session"):          # no artificial rows
        gen([holiday], [date(2024, 7, 4)])
    with pytest.raises(MarketDataError, match="not a trading session"):
        gen([holiday], [date(2024, 7, 6)])                                        # Saturday


def test_dst_transition_uses_calendar_timestamps():
    # 2024-03-08 (EST): row at 21:30 UTC.  2024-03-11 (EDT): row at 20:30 UTC.
    in_est = scored(1, datetime(2024, 3, 8, 21, 0, tzinfo=UTC))                 # 16:00 EST -> eligible 03-08
    after_edt_row = scored(2, datetime(2024, 3, 11, 20, 45, tzinfo=UTC))         # 16:45 EDT -> not in 03-11
    rows = gen([in_est, after_edt_row], [date(2024, 3, 8), date(2024, 3, 11), date(2024, 3, 12)])
    r8, r11, r12 = (one(rows, day=date(2024, 3, d)) for d in (8, 11, 12))
    assert r8.prediction_timestamp == datetime(2024, 3, 8, 21, 30, tzinfo=UTC) and r8.news_count == 1
    assert r11.prediction_timestamp == datetime(2024, 3, 11, 20, 30, tzinfo=UTC) and r11.news_count == 1
    assert r12.news_count == 2


def test_timestamps_are_timezone_aware_utc():
    rows = gen([], [D, date(2024, 3, 8)])
    for r in rows:
        assert r.prediction_timestamp.tzinfo == UTC
    frame = features_to_frame(rows)
    assert str(frame["prediction_timestamp"].dt.tz) == "UTC"


# ==========================================
# No look-ahead (mandatory)
# ==========================================

DATES = [date(2024, 7, d) for d in (8, 9, 10, 11, 12, 15)]


def _by_date(rows):
    return {r.trading_date: r for r in rows}


def test_A_moving_an_article_later_changes_only_rows_from_its_new_or_old_time_on():
    base = [scored(i, ny(2024, 7, 8 + (i % 4), 9 + i, 0), [POS, NEG, NEU][i % 3]) for i in range(8)]
    moved = [s if s.article.provider_article_id != "2" else
             scored(2, s.article.created_at, [POS, NEG, NEU][2], updated=ny(2024, 7, 12, 10, 0)) for s in base]
    before, after = _by_date(gen(base, DATES)), _by_date(gen(moved, DATES))
    original_day = base[2].article.information_available_at.astimezone(NY).date()      # 2024-07-10
    for d in DATES:
        if d < original_day:
            assert before[d] == after[d]                                           # earlier rows untouched
    # rows between the old and new availability lose the article; from 07-12 on it is back
    for d in (date(2024, 7, 10), date(2024, 7, 11)):
        assert after[d].news_count == before[d].news_count - 1
    for d in (date(2024, 7, 12), date(2024, 7, 15)):
        assert after[d].news_count == before[d].news_count


def test_B_post_market_article_does_not_touch_same_day_row():
    early = gen([scored(1, ny(2024, 7, 10, 11, 0))], DATES)
    with_post = gen([scored(1, ny(2024, 7, 10, 11, 0)), scored(2, ny(2024, 7, 10, 17, 0), NEG)], DATES)
    e, w = _by_date(early), _by_date(with_post)
    assert w[date(2024, 7, 10)] == e[date(2024, 7, 10)]
    assert w[date(2024, 7, 11)] != e[date(2024, 7, 11)]


def test_C_boundary_equality_contributes():
    rows = _by_date(gen([scored(1, ny(2024, 7, 10, 16, 30))], DATES))
    assert rows[date(2024, 7, 9)].news_count == 0 and rows[date(2024, 7, 10)].news_count == 1


def test_D_mutating_an_article_into_the_future_leaves_earlier_rows_unchanged():
    items = [scored(1, ny(2024, 7, 8, 10, 0)), scored(2, ny(2024, 7, 9, 10, 0), NEG),
             scored(3, ny(2024, 7, 11, 10, 0), NEU)]
    mutated = items[:2] + [scored(3, ny(2024, 7, 11, 10, 0), NEU, updated=ny(2024, 7, 30, 9, 0))]
    before, after = _by_date(gen(items, DATES)), _by_date(gen(mutated, DATES))
    for d in (date(2024, 7, 8), date(2024, 7, 9), date(2024, 7, 10)):
        assert before[d] == after[d]
    assert after[date(2024, 7, 11)].news_count == before[date(2024, 7, 11)].news_count - 1


@pytest.mark.parametrize("horizon_pair", [(1, 5)])
def test_E_horizon_invariance(horizon_pair):
    sessions = [d for d in pd.bdate_range("2024-07-01", "2024-07-31").date if CAL.is_session(d)]
    frame = pd.DataFrame({"Date": pd.to_datetime(sessions), "Close": np.linspace(100, 110, len(sessions))})
    items = [scored(i, ny(2024, 7, 1) + timedelta(hours=7 * i), [POS, NEG, NEU][i % 3]) for i in range(80)]

    results = {}
    for h in horizon_pair:
        row_dates = build_targets(frame, horizon=h)["Date"]                      # rows usable for horizon h
        results[h] = _by_date(gen(items, row_dates))
    shared = set(results[1]) & set(results[5])
    assert shared and all(results[1][d] == results[5][d] for d in shared)


def test_no_row_ever_uses_an_article_after_its_prediction_timestamp():
    rng = random.Random(3)
    items = [scored(i, ny(2024, 7, 1) + timedelta(minutes=rng.randrange(0, 60 * 24 * 30)),
                    [POS, NEG, NEU][i % 3]) for i in range(150)]
    sessions = [d for d in pd.bdate_range("2024-07-01", "2024-07-31").date if CAL.is_session(d)]
    for row in gen(items, sessions):
        expected = sum(s.article.information_available_at <= row.prediction_timestamp for s in items)
        assert row.news_count == expected


# ==========================================
# Determinism & chronological single pass
# ==========================================


def test_shuffled_input_gives_identical_output():
    rng = random.Random(7)
    items = [scored(i, ny(2024, 7, 8) + timedelta(hours=5 * i), [POS, NEG, NEU][i % 3],
                    symbols=[("AAPL",), ("AAPL", "MSFT"), ("MSFT",)][i % 3]) for i in range(30)]
    reference = gen(items, DATES, symbols=["AAPL", "MSFT"])
    for _ in range(10):
        shuffled_items, shuffled_dates = items[:], DATES[:]
        rng.shuffle(shuffled_items)
        rng.shuffle(shuffled_dates)
        assert gen(shuffled_items, shuffled_dates, symbols=["MSFT", "AAPL"]) == reference
    assert [(r.symbol, r.trading_date) for r in reference] == sorted((r.symbol, r.trading_date) for r in reference)


def test_single_pass_matches_direct_computation_for_every_row():
    items = [scored(i, ny(2024, 7, 8) + timedelta(hours=3 * i), [POS, NEG, NEU][i % 3]) for i in range(40)]
    for row in gen(items, DATES):
        assert features_as_of(items, row.trading_date, CAL, "AAPL") == row


def test_duplicate_trading_dates_rejected_and_empty_dates_ok():
    with pytest.raises(SentimentFeatureError, match="duplicates"):
        gen([], [D, D])
    assert gen([scored(1, ny(2024, 7, 10, 10, 0))], []) == []


# ==========================================
# Output schema
# ==========================================


def test_output_schema():
    rows = gen([scored(1, ny(2024, 7, 10, 10, 0))], [D])
    row = rows[0]
    assert tuple(row.to_dict()) == OUTPUT_COLUMNS
    assert OUTPUT_COLUMNS[:4] == ("symbol", "trading_date", "prediction_timestamp", "sentiment_feature_version")
    assert OUTPUT_COLUMNS[4:] == FEATURE_COLUMNS and len(FEATURE_COLUMNS) == 15
    assert row.sentiment_feature_version == SENTIMENT_FEATURE_VERSION == "sentiment_features_v1"
    assert row.trading_date == D and isinstance(row.trading_date, date)
    frame = features_to_frame(rows)
    assert list(frame.columns) == list(OUTPUT_COLUMNS) and len(frame) == 1
