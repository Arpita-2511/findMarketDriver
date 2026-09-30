"""
News temporal alignment and leakage protection - Phase 4B.

Every scenario is written in New York time for readability; all
comparisons happen on UTC-aware timestamps.
"""

import random
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from services.market_calendar_service import TradingCalendar, completion_time
from services.news_alignment import (
    FutureInformationError,
    SessionPhase,
    assign_session,
    assign_sessions,
    classify_phase,
    eligible_articles,
    is_eligible,
    row_prediction_timestamp,
    to_exchange_time,
)
from services.market_data import MarketDataError
from services.news_schema import NewsArticle, historical_information_available_at
from tests.conftest import calendar_2024
from training.evaluation_harness import build_targets

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
FETCHED = datetime(2026, 10, 1, tzinfo=UTC)
CAL = calendar_2024()


def ny(*args) -> datetime:
    return datetime(*args, tzinfo=NY)


def news(article_id, created, updated=None, symbols=("AAPL",), headline=None, fetched=FETCHED) -> NewsArticle:
    """Built exactly like ingestion does: availability from the historical rule."""
    return NewsArticle(
        provider="alpaca", provider_article_id=str(article_id),
        headline=headline or f"News {article_id}", symbols=symbols,
        created_at=created, updated_at=updated,
        information_available_at=historical_information_available_at(created, updated),
        fetched_at=fetched,
    )


def keys(articles):
    return [a.article_key for a in articles]


# ==========================================
# 1-3, 20. Eligibility boundary
# ==========================================


@pytest.mark.parametrize("article_time, prediction, expected", [
    (ny(2024, 7, 10, 10, 30), ny(2024, 7, 10, 9, 30), False),   # example 1
    (ny(2024, 7, 10, 10, 30), ny(2024, 7, 10, 11, 0), True),    # example 2
    (ny(2024, 7, 10, 17, 30), ny(2024, 7, 10, 16, 0), False),   # example 3
    (ny(2024, 7, 10, 17, 30), ny(2024, 7, 11, 9, 30), True),    # example 4 (next session)
])
def test_prompt_examples(article_time, prediction, expected):
    assert is_eligible(news(1, article_time), prediction) is expected


def test_before_equal_after():
    p = ny(2024, 7, 10, 12, 0)
    assert is_eligible(news(1, p - timedelta(seconds=1)), p)
    assert is_eligible(news(2, p), p)                              # exactly at the boundary
    assert not is_eligible(news(3, p + timedelta(seconds=1)), p)


def test_several_articles_around_the_boundary():
    p = ny(2024, 7, 10, 16, 30)                                    # a direction_v1 row's prediction time
    arts = [news(i, p + timedelta(seconds=s)) for i, s in enumerate([-3600, -1, 0, 1, 3600])]
    assert keys(eligible_articles(arts, p)) == ["alpaca:0", "alpaca:1", "alpaca:2"]


# ==========================================
# 4-10. Session phases, weekends, holidays
# ==========================================


@pytest.mark.parametrize("local, phase, session", [
    (ny(2024, 7, 10, 7, 0), SessionPhase.PRE_MARKET, date(2024, 7, 10)),
    (ny(2024, 7, 10, 9, 29, 59), SessionPhase.PRE_MARKET, date(2024, 7, 10)),
    (ny(2024, 7, 10, 9, 30), SessionPhase.REGULAR, date(2024, 7, 10)),
    (ny(2024, 7, 10, 13, 0), SessionPhase.REGULAR, date(2024, 7, 10)),
    (ny(2024, 7, 10, 16, 0), SessionPhase.REGULAR, date(2024, 7, 10)),        # cutoff inclusive
    (ny(2024, 7, 10, 16, 0, 1), SessionPhase.POST_MARKET, date(2024, 7, 11)),
    (ny(2024, 7, 10, 23, 59), SessionPhase.POST_MARKET, date(2024, 7, 11)),
    (ny(2024, 7, 12, 18, 0), SessionPhase.POST_MARKET, date(2024, 7, 15)),   # Friday -> Monday
    (ny(2024, 7, 13, 11, 0), SessionPhase.NON_TRADING_DAY, date(2024, 7, 15)),  # Saturday
    (ny(2024, 7, 14, 20, 0), SessionPhase.NON_TRADING_DAY, date(2024, 7, 15)),  # Sunday
    (ny(2024, 7, 4, 10, 0), SessionPhase.NON_TRADING_DAY, date(2024, 7, 5)),    # July 4 holiday
    (ny(2024, 7, 3, 17, 0), SessionPhase.POST_MARKET, date(2024, 7, 5)),     # evening before holiday
    (ny(2024, 5, 24, 17, 0), SessionPhase.POST_MARKET, date(2024, 5, 28)),   # Fri before Memorial Day
])
def test_session_assignment(local, phase, session):
    a = assign_session(news(1, local), CAL)
    assert (a.phase, a.session_date) == (phase, session)
    assert a.calendar_date == local.date()


def test_session_assignment_does_not_grant_eligibility():
    """10:30 news falls INTO the 07-10 session but must not reach that day's 09:30 prediction."""
    article = news(1, ny(2024, 7, 10, 10, 30))
    assert assign_session(article, CAL).session_date == date(2024, 7, 10)
    assert not is_eligible(article, CAL.session_open(date(2024, 7, 10)))


def test_article_after_last_known_session_is_not_guessed():
    with pytest.raises(MarketDataError):
        assign_session(news(1, ny(2024, 7, 31, 17, 0)), CAL)      # next session unknown


# ==========================================
# 11-13. Timezones, DST, naive datetimes
# ==========================================


def test_timezone_conversion_keeps_utc_internally():
    a = assign_session(news(1, datetime(2024, 7, 10, 14, 30, tzinfo=UTC)), CAL)
    assert a.information_available_at.tzinfo == UTC
    assert a.exchange_local_time.hour == 10 and a.exchange_local_time.minute == 30   # EDT = UTC-4
    assert a.phase is SessionPhase.REGULAR


def test_dst_transition_changes_local_phase_for_same_utc_clock():
    # 2024-03-10: clocks spring forward. 13:30 UTC is 08:30 EST before, 09:30 EDT after.
    before = assign_session(news(1, datetime(2024, 3, 8, 13, 30, tzinfo=UTC)), CAL)
    after = assign_session(news(2, datetime(2024, 3, 11, 13, 30, tzinfo=UTC)), CAL)
    assert before.phase is SessionPhase.PRE_MARKET and before.exchange_local_time.hour == 8
    assert after.phase is SessionPhase.REGULAR and after.exchange_local_time.hour == 9
    # post-market Friday in EST rolls to Monday in EDT
    friday_evening = assign_session(news(3, datetime(2024, 3, 8, 21, 30, tzinfo=UTC)), CAL)
    assert (friday_evening.phase, friday_evening.session_date) == (SessionPhase.POST_MARKET, date(2024, 3, 11))
    # row prediction timestamps follow DST too (16:30 local)
    assert row_prediction_timestamp(date(2024, 3, 8), CAL) == datetime(2024, 3, 8, 21, 30, tzinfo=UTC)
    assert row_prediction_timestamp(date(2024, 3, 11), CAL) == datetime(2024, 3, 11, 20, 30, tzinfo=UTC)


def test_classify_phase_reads_new_york_clock_and_rejects_naive():
    """Review fix: 13:00 UTC is 09:00 EDT (pre-market), not 13:00 regular session."""
    assert classify_phase(datetime(2024, 7, 10, 13, 0, tzinfo=UTC), CAL) is SessionPhase.PRE_MARKET
    assert classify_phase(ny(2024, 7, 10, 9, 0), CAL) is SessionPhase.PRE_MARKET
    with pytest.raises(ValueError, match="timezone-aware"):
        classify_phase(datetime(2024, 7, 10, 13, 0), CAL)


def test_november_dst_transition():
    # 2024-11-03: clocks fall back. 13:30 UTC is 09:30 EDT on Fri 11-01, 08:30 EST on Mon 11-04.
    cal = TradingCalendar(d.date() for d in pd.bdate_range("2024-10-28", "2024-11-08"))
    friday = assign_session(news(1, datetime(2024, 11, 1, 13, 30, tzinfo=UTC)), cal)
    monday = assign_session(news(2, datetime(2024, 11, 4, 13, 30, tzinfo=UTC)), cal)
    assert (friday.phase, friday.exchange_local_time.hour) == (SessionPhase.REGULAR, 9)
    assert (monday.phase, monday.exchange_local_time.hour) == (SessionPhase.PRE_MARKET, 8)
    assert row_prediction_timestamp(date(2024, 11, 1), cal) == datetime(2024, 11, 1, 20, 30, tzinfo=UTC)
    assert row_prediction_timestamp(date(2024, 11, 4), cal) == datetime(2024, 11, 4, 21, 30, tzinfo=UTC)


def test_naive_datetimes_rejected():
    naive = datetime(2024, 7, 10, 12, 0)
    article = news(1, ny(2024, 7, 10, 10, 0))
    with pytest.raises(ValueError, match="timezone-aware"):
        is_eligible(article, naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        eligible_articles([article], ny(2024, 7, 10, 12, 0), not_before=naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        to_exchange_time(naive)


# ==========================================
# 14. Future information
# ==========================================


def test_article_available_after_it_was_fetched_is_rejected():
    impossible = news(1, ny(2024, 7, 10, 10, 0), fetched=ny(2024, 7, 10, 9, 0))
    with pytest.raises(FutureInformationError):
        eligible_articles([impossible], ny(2024, 7, 11, 10, 0))
    with pytest.raises(FutureInformationError):
        assign_session(impossible, CAL)
    with pytest.raises(FutureInformationError):          # review fix: single-article check too
        is_eligible(impossible, ny(2024, 7, 11, 10, 0))


def test_live_prediction_in_the_future_is_rejected():
    now = ny(2024, 7, 10, 12, 0)
    with pytest.raises(FutureInformationError, match="in the future"):
        eligible_articles([], now + timedelta(minutes=1), now=now)
    assert eligible_articles([], now, now=now) == []


# ==========================================
# 15-16. information_available_at governs; created_at is preserved
# ==========================================


def test_information_available_at_not_created_at_controls_eligibility():
    # created 11:00, revised 12:00 -> the stored version is available from 12:00
    revised = news(1, ny(2024, 7, 10, 11, 0), updated=ny(2024, 7, 10, 12, 0))
    assert not is_eligible(revised, ny(2024, 7, 10, 11, 30))
    assert is_eligible(revised, ny(2024, 7, 10, 12, 0))


def test_updated_at_moves_session_but_never_replaces_created_at():
    revised = news(1, ny(2024, 7, 10, 15, 0), updated=ny(2024, 7, 10, 17, 0))
    a = assign_session(revised, CAL)
    assert a.phase is SessionPhase.POST_MARKET and a.session_date == date(2024, 7, 11)
    assert a.created_at == ny(2024, 7, 10, 15, 0) == revised.created_at


# ==========================================
# 17. Leakage regression - explicit
# ==========================================


BASE = [news(1, ny(2024, 7, 9, 18, 0)), news(2, ny(2024, 7, 10, 10, 0)),
        news(3, ny(2024, 7, 10, 16, 45)), news(4, ny(2024, 7, 11, 9, 0))]
PREDICTION = ny(2024, 7, 10, 16, 30)       # row for 2024-07-10


def test_leakage_news_after_prediction_cannot_change_earlier_prediction():
    before = eligible_articles(BASE, PREDICTION)
    assert keys(before) == ["alpaca:1", "alpaca:2"]

    later_news = BASE + [news(5, PREDICTION + timedelta(seconds=1)), news(6, ny(2024, 7, 20, 9, 0))]
    edited_future = [a if a.provider_article_id != "3" else
                     news(3, a.created_at, updated=ny(2024, 7, 12, 8, 0), headline="Edited later") for a in BASE]

    assert eligible_articles(later_news, PREDICTION) == before
    assert eligible_articles(edited_future, PREDICTION) == before


def test_leakage_inverse_news_before_prediction_is_included():
    early = news(7, PREDICTION - timedelta(minutes=1))
    after = eligible_articles(BASE + [early], PREDICTION)
    assert keys(after) == ["alpaca:1", "alpaca:2", "alpaca:7"]


def test_no_prediction_ever_consumes_a_future_article():
    rng = random.Random(0)
    arts = [news(i, ny(2024, 7, 1) + timedelta(minutes=rng.randrange(0, 60 * 24 * 25))) for i in range(300)]
    for session in [d for d in pd.bdate_range("2024-07-01", "2024-07-25").date if CAL.is_session(d)]:
        p = row_prediction_timestamp(session, CAL)
        chosen = eligible_articles(arts, p)
        assert all(a.information_available_at <= p for a in chosen)
        assert len(chosen) == sum(a.information_available_at <= p for a in arts)


def test_lookback_window_is_exclusive_below_inclusive_above():
    lower = ny(2024, 7, 10, 10, 0)
    assert keys(eligible_articles(BASE, PREDICTION, not_before=lower)) == []       # article 2 == lower -> excluded
    assert keys(eligible_articles(BASE, PREDICTION, not_before=lower - timedelta(seconds=1))) == ["alpaca:2"]
    with pytest.raises(ValueError, match="earlier"):
        eligible_articles(BASE, PREDICTION, not_before=PREDICTION)


# ==========================================
# 18-19. Determinism and empty input
# ==========================================


def test_deterministic_regardless_of_input_order():
    same_time = ny(2024, 7, 10, 10, 0)
    arts = BASE + [news(10, same_time), news(9, same_time)]
    shuffled = list(reversed(arts))
    assert eligible_articles(arts, PREDICTION) == eligible_articles(shuffled, PREDICTION)
    # ties at 10:00 break on (provider, provider_article_id) - string order "10" < "2" < "9"
    assert keys(eligible_articles(arts, PREDICTION)) == ["alpaca:1", "alpaca:10", "alpaca:2", "alpaca:9"]
    assert assign_sessions(arts, CAL) == assign_sessions(shuffled, CAL)


def test_empty_input():
    assert eligible_articles([], PREDICTION) == []
    assert assign_sessions([], CAL) == []


# ==========================================
# 21. Multiple symbols, no cross-ticker leakage
# ==========================================


def test_multi_symbol_articles_reach_only_their_listed_symbols():
    both = news(1, ny(2024, 7, 10, 10, 0), symbols=("AAPL", "MSFT"))
    msft_only = news(2, ny(2024, 7, 10, 11, 0), symbols=("MSFT",))
    arts = [both, msft_only]

    assert keys(eligible_articles(arts, PREDICTION, symbol="AAPL")) == ["alpaca:1"]
    assert keys(eligible_articles(arts, PREDICTION, symbol="msft")) == ["alpaca:1", "alpaca:2"]
    assert eligible_articles(arts, PREDICTION, symbol="GOOG") == []
    assert assign_session(both, CAL).symbols == ("AAPL", "MSFT")        # never invented or dropped


# ==========================================
# 22. direction_v1 rows and prediction horizons
# ==========================================


@pytest.mark.parametrize("horizon", [1, 5])
def test_horizon_never_moves_the_prediction_timestamp(horizon):
    sessions = [d for d in pd.bdate_range("2024-07-01", "2024-07-31").date if CAL.is_session(d)]
    frame = pd.DataFrame({"Date": pd.to_datetime(sessions), "Close": np.linspace(100, 120, len(sessions))})
    rows = build_targets(frame, horizon=horizon)

    t = rows.index[rows["Date"] == pd.Timestamp("2024-07-10")][0]
    label_session = sessions[t + horizon]                          # Close[t+h] used by the label
    in_label_window = news(1, ny(2024, 7, 11, 10, 0))              # between t's close and t+h's close
    before_row = news(2, ny(2024, 7, 10, 15, 0))

    p = row_prediction_timestamp(rows.loc[t, "Date"], CAL)
    assert p == completion_time(date(2024, 7, 10)).astimezone(UTC)
    assert label_session > date(2024, 7, 10)
    assert keys(eligible_articles([in_label_window, before_row], p)) == ["alpaca:2"]


def test_row_prediction_timestamp_requires_a_session():
    with pytest.raises(MarketDataError, match="not a trading session"):
        row_prediction_timestamp(date(2024, 7, 4), CAL)
