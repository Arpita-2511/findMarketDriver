"""sentiment_window_v1: 1/5/20-session windows, cutoff, coverage, empty windows, determinism."""

import math
import random
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from services.market_calendar_service import TradingCalendar
from services.market_data import MarketDataError
from services.news_alignment import row_prediction_timestamp
from services.sentiment_features import ScoredArticle
from services.sentiment_window import COLUMNS, SentimentWindowError, generate_sentiment_window_features
from tests.event_fakes import NEG, NEU, POS, sentiment_for
from tests.sentiment_fakes import news

NY = ZoneInfo("America/New_York")
HOLIDAYS = {date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2)}
SESSIONS = [d for d in pd.bdate_range("2024-06-03", "2024-09-30").date if d not in HOLIDAYS]
CAL = TradingCalendar(SESSIONS)
START = datetime(2024, 6, 3, tzinfo=NY)                 # news interval start (first session is covered from here)
END = datetime(2024, 12, 31, tzinfo=timezone.utc)
FIRST_COVERED = SESSIONS[20]                            # prev_20(D) must be >= the first in-interval session


def scored(i, when, probs=NEU, symbols=("AAPL",)):
    a = news(i, when, symbols=symbols)
    return ScoredArticle(a, sentiment_for(a, probs))


def ts(d):
    return row_prediction_timestamp(d, CAL)


def build(items, dates=SESSIONS, start=START, end=END):
    return generate_sentiment_window_features(items, dates, CAL, "AAPL", coverage_start=start, coverage_end=end)


def row(frame, d):
    return frame.set_index("trading_date").loc[d]


def test_coverage_requires_the_full_20_session_window():
    f = build([])
    first = f[f["covered"]]["trading_date"].iloc[0]
    assert first == FIRST_COVERED
    early = f[~f["covered"]]
    assert len(early) == 20 and early[list(COLUMNS)].isna().all().all()     # unknown, not "no news"
    late = build([], end=datetime(2024, 9, 1, tzinfo=timezone.utc))
    assert not row(late, date(2024, 9, 3))["covered"]                        # ts(D) after the interval end


def test_one_session_window_boundaries():
    d = date(2024, 7, 15)
    prev = CAL.previous_session_before(d, 1)
    items = [scored(1, ts(prev)),                                  # exactly ts(prev): belongs to prev, not D
             scored(2, ts(prev) + timedelta(seconds=1)),           # first moment of D's window
             scored(3, ts(d)),                                     # exactly the cutoff: included (<=)
             scored(4, ts(d) + timedelta(seconds=1))]              # after the cutoff: next session
    f = build(items)
    assert row(f, d)["sw_count_1"] == 2
    assert row(f, prev)["sw_count_1"] == 1
    assert row(f, CAL.next_session_after(d))["sw_count_1"] == 1


def test_weekend_news_reaches_the_next_session_only():
    f = build([scored(1, datetime(2024, 7, 13, 12, 0, tzinfo=NY))])          # Saturday
    assert row(f, date(2024, 7, 12))["sw_count_1"] == 0
    assert row(f, date(2024, 7, 15))["sw_count_1"] == 1


def test_5_and_20_session_window_counts():
    items = [scored(i, datetime.combine(d, datetime.min.time(), NY) + timedelta(hours=12))
             for i, d in enumerate(SESSIONS)]                       # one article per session, mid-day
    f = build(items)
    for d in SESSIONS[25:]:
        r = row(f, d)
        assert (r["sw_count_1"], r["sw_count_5"], r["sw_count_20"]) == (1, 5, 20)
        assert r["sw_count_surprise_1_20"] == pytest.approx(1 - 20 / 20)


def test_means_ratios_change_and_surprise():
    d = date(2024, 8, 15)
    s = [CAL.previous_session_before(d, k) for k in range(1, 20)]
    at = lambda day, h=12: datetime.combine(day, datetime.min.time(), NY) + timedelta(hours=h)  # noqa: E731
    items = [scored(1, at(d), POS), scored(2, at(d, 13), NEG),           # D: +0.6, -0.7
             scored(3, at(s[2]), POS),                                    # within 5 sessions: +0.6
             scored(4, at(s[10]), NEU), scored(5, at(s[18]), NEG)]        # within 20 only: +0.1, -0.7
    r = row(build(items), d)
    m1, m5, m20 = (0.6 - 0.7) / 2, (0.6 - 0.7 + 0.6) / 3, (0.6 - 0.7 + 0.6 + 0.1 - 0.7) / 5
    assert (r["sw_count_1"], r["sw_count_5"], r["sw_count_20"]) == (2, 3, 5)
    assert r["sw_mean_1"] == pytest.approx(m1) and r["sw_mean_5"] == pytest.approx(m5)
    assert r["sw_mean_20"] == pytest.approx(m20)
    assert r["sw_positive_ratio_5"] == pytest.approx(2 / 3) and r["sw_negative_ratio_5"] == pytest.approx(1 / 3)
    assert r["sw_positive_ratio_20"] == pytest.approx(2 / 5) and r["sw_negative_ratio_20"] == pytest.approx(2 / 5)
    assert r["sw_mean_change_5_20"] == pytest.approx(m5 - m20)
    assert r["sw_mean_surprise_1_20"] == pytest.approx(m1 - m20)
    assert r["sw_count_surprise_1_20"] == pytest.approx(2 - 5 / 20)


def test_empty_window_convention():
    r = row(build([]), date(2024, 8, 15))
    assert r["covered"]
    for c in COLUMNS:
        assert r[c] == 0.0 and not math.isnan(r[c])


def test_future_articles_never_change_earlier_rows():
    d = date(2024, 8, 15)
    base = [scored(1, datetime(2024, 8, 14, 10, tzinfo=NY), POS)]
    future = base + [scored(2, ts(d) + timedelta(seconds=1), NEG), scored(3, datetime(2024, 9, 20, 9, tzinfo=NY), NEG)]
    a, b = build(base), build(future)
    upto = a["trading_date"] <= d
    pd.testing.assert_frame_equal(a[upto], b[upto])


def test_other_symbols_are_ignored():
    f = build([scored(1, datetime(2024, 8, 15, 10, tzinfo=NY), POS, symbols=("MSFT",))])
    assert row(f, date(2024, 8, 15))["sw_count_1"] == 0


def test_deterministic_and_input_order_independent():
    items = [scored(i, datetime(2024, 7, 1, 9, tzinfo=NY) + timedelta(hours=7 * i), (POS, NEG, NEU)[i % 3])
             for i in range(200)]
    shuffled = items[:]
    random.Random(7).shuffle(shuffled)
    pd.testing.assert_frame_equal(build(items), build(items))
    pd.testing.assert_frame_equal(build(items), build(shuffled, dates=list(reversed(SESSIONS))))


def test_invalid_dates_raise():
    with pytest.raises(SentimentWindowError):
        build([], dates=[SESSIONS[30], SESSIONS[30]])
    with pytest.raises(MarketDataError):
        build([], dates=[date(2024, 7, 4)])                                  # holiday: not a session
    with pytest.raises(ValueError):
        build([], start=datetime(2024, 6, 3))                                # naive coverage bound
