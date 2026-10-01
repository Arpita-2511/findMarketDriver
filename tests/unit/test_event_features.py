"""Phase 6: leakage-safe daily event features (event_features_v1), offline."""

import math
import random
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from services.event_features import (
    EVENT_FEATURE_COLUMNS,
    EVENT_OUTPUT_COLUMNS,
    EventFeatureError,
    generate_event_features,
)
from services.market_data import MarketDataError
from tests.conftest import calendar_2024
from tests.event_fakes import NEG, NEU, POS, classified
from training.evaluation_harness import build_targets

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
CAL = calendar_2024()

EARN, LEGAL, ANALYST = "Apple Reports Q1 EPS beat", "Apple Says It Is Suing Qualcomm", "Wells Fargo Cuts Apple Estimates"
NOISE = "Inside The Hidden Economy Of Pawn Shops"              # OTHER


def ny(*args):
    return datetime(*args, tzinfo=NY)


def rows_by_date(items, dates, symbols=("AAPL",)):
    rows = generate_event_features(items, dates, CAL, symbols)
    return {(r["symbol"], r["trading_date"]): r for r in rows}


def row(items, d, symbol="AAPL"):
    return rows_by_date(items, [d], [symbol])[(symbol, d)]


D = date(2024, 7, 10)


# ==========================================
# Counts, windows, zero rows
# ==========================================


def test_daily_counts_and_session_window():
    items = [classified(1, ny(2024, 7, 10, 10, 0), EARN, POS), classified(2, ny(2024, 7, 10, 11, 0), LEGAL, NEG),
             classified(3, ny(2024, 7, 10, 12, 0), NOISE), classified(4, ny(2024, 7, 10, 17, 0), ANALYST)]
    r = rows_by_date(items, [date(2024, 7, 10), date(2024, 7, 11)])
    d10, d11 = r[("AAPL", date(2024, 7, 10))], r[("AAPL", date(2024, 7, 11))]
    assert (d10["article_count"], d10["event_count"], d10["unique_event_type_count"]) == (3, 2, 2)
    assert (d10["earnings_event_count"], d10["legal_event_count"], d10["analyst_rating_event_count"]) == (1, 1, 0)
    assert (d10["positive_event_count"], d10["negative_event_count"], d10["neutral_event_count"]) == (1, 1, 0)
    assert d10["event_impact_score"] == pytest.approx((0.6 + -0.7) / 2)           # OTHER excluded
    assert (d11["article_count"], d11["analyst_rating_event_count"]) == (1, 1)       # 17:00 -> next session


def test_every_article_lands_in_exactly_one_row():
    rng = random.Random(1)
    items = [classified(i, ny(2024, 7, 1) + timedelta(minutes=rng.randrange(0, 60 * 24 * 20)),
                        [EARN, LEGAL, ANALYST, NOISE][i % 4]) for i in range(80)]
    sessions = [d for d in pd.bdate_range("2024-07-01", "2024-07-31").date if CAL.is_session(d)]
    rows = generate_event_features(items, sessions, CAL, ["AAPL"])
    last_ts = rows[-1]["prediction_timestamp"]
    first_lower = CAL.previous_session_before(sessions[0])
    assert sum(r["article_count"] for r in rows) == sum(
        1 for c in items if c.article.information_available_at <= last_ts)
    assert first_lower is not None                                               # window bound exists


def test_zero_event_day_is_all_zero_and_none():
    r = row([classified(1, ny(2024, 7, 10, 12, 0), NOISE)], D)                   # only OTHER
    assert r["article_count"] == 1 and r["event_count"] == 0
    for name in EVENT_FEATURE_COLUMNS:
        if name not in ("article_count", "dominant_event_type"):
            assert r[name] == 0 and not (isinstance(r[name], float) and math.isnan(r[name]))
    assert r["dominant_event_type"] == "NONE"
    assert row([], D)["article_count"] == 0


def test_dominant_type_and_confidence_mean():
    items = [classified(1, ny(2024, 7, 10, 9, 0), LEGAL), classified(2, ny(2024, 7, 10, 10, 0), LEGAL + " again"),
             classified(3, ny(2024, 7, 10, 11, 0), EARN)]
    r = row(items, D)
    assert r["dominant_event_type"] == "LEGAL"
    assert r["mean_event_confidence"] == pytest.approx(np.mean([c.event_confidence for c in items]))
    tie = row([classified(1, ny(2024, 7, 10, 9, 0), LEGAL), classified(2, ny(2024, 7, 10, 10, 0), EARN)], D)
    assert tie["dominant_event_type"] == "EARNINGS"                                # taxonomy order breaks ties


def test_recent_event_count_spans_five_sessions():
    # sessions before 07-10: 07-09, 07-08, 07-05, 07-03, 07-02 -> lower bound ts(07-02)
    items = [classified(1, ny(2024, 7, 2, 12, 0), EARN), classified(2, ny(2024, 7, 2, 17, 0), EARN + " q2"),
             classified(3, ny(2024, 7, 9, 12, 0), LEGAL), classified(4, ny(2024, 7, 10, 12, 0), ANALYST)]
    r = row(items, D)
    assert r["event_count"] == 1 and r["recent_event_count"] == 3                 # 07-02 12:00 is outside


# ==========================================
# Leakage & timing
# ==========================================


def test_boundary_equality_included_and_later_excluded():
    items = [classified(1, ny(2024, 7, 10, 16, 30), EARN), classified(2, ny(2024, 7, 10, 16, 30, 1), LEGAL)]
    r = rows_by_date(items, [date(2024, 7, 10), date(2024, 7, 11)])
    assert r[("AAPL", date(2024, 7, 10))]["earnings_event_count"] == 1
    assert r[("AAPL", date(2024, 7, 10))]["legal_event_count"] == 0
    assert r[("AAPL", date(2024, 7, 11))]["legal_event_count"] == 1


def test_future_article_cannot_change_earlier_rows():
    base = [classified(i, ny(2024, 7, 8 + i, 10, 0), [EARN, LEGAL, ANALYST][i]) for i in range(3)]
    future = base + [classified(9, ny(2024, 7, 12, 10, 0), EARN)]
    dates = [date(2024, 7, d) for d in (8, 9, 10, 11)]
    assert rows_by_date(base, dates) == rows_by_date(future, dates)


def test_availability_not_creation_controls_the_row():
    revised = classified(1, ny(2024, 7, 10, 10, 0), EARN, updated=ny(2024, 7, 10, 18, 0))
    r = rows_by_date([revised], [date(2024, 7, 10), date(2024, 7, 11)])
    assert r[("AAPL", date(2024, 7, 10))]["event_count"] == 0
    assert r[("AAPL", date(2024, 7, 11))]["event_count"] == 1


def test_weekend_and_holiday_alignment():
    items = [classified(1, ny(2024, 7, 13, 11, 0), EARN), classified(2, ny(2024, 7, 4, 12, 0), LEGAL)]
    r = rows_by_date(items, [date(2024, 7, 3), date(2024, 7, 5), date(2024, 7, 12), date(2024, 7, 15)])
    assert r[("AAPL", date(2024, 7, 3))]["event_count"] == 0
    assert r[("AAPL", date(2024, 7, 5))]["legal_event_count"] == 1
    assert r[("AAPL", date(2024, 7, 12))]["event_count"] == 0
    assert r[("AAPL", date(2024, 7, 15))]["earnings_event_count"] == 1
    with pytest.raises(MarketDataError, match="not a trading session"):
        generate_event_features(items, [date(2024, 7, 4)], CAL, ["AAPL"])


def test_dst_and_timezone_aware_output():
    items = [classified(1, datetime(2024, 3, 8, 21, 0, tzinfo=UTC), EARN),       # 16:00 EST
             classified(2, datetime(2024, 3, 11, 20, 45, tzinfo=UTC), LEGAL)]     # 16:45 EDT
    r = rows_by_date(items, [date(2024, 3, 8), date(2024, 3, 11), date(2024, 3, 12)])
    assert r[("AAPL", date(2024, 3, 8))]["prediction_timestamp"] == datetime(2024, 3, 8, 21, 30, tzinfo=UTC)
    assert r[("AAPL", date(2024, 3, 11))]["prediction_timestamp"] == datetime(2024, 3, 11, 20, 30, tzinfo=UTC)
    assert r[("AAPL", date(2024, 3, 8))]["event_count"] == 1
    assert r[("AAPL", date(2024, 3, 11))]["event_count"] == 0
    assert r[("AAPL", date(2024, 3, 12))]["event_count"] == 1
    assert all(x["prediction_timestamp"].tzinfo == UTC for x in r.values())


def test_first_calendar_session_has_no_lower_bound():
    early = classified(1, datetime(2024, 2, 20, 15, 0, tzinfo=UTC), EARN)         # before the calendar starts
    assert CAL.previous_session_before(date(2024, 3, 1)) is None
    assert row([early], date(2024, 3, 1))["event_count"] == 1


@pytest.mark.parametrize("pair", [(1, 5)])
def test_horizon_invariance(pair):
    sessions = [d for d in pd.bdate_range("2024-07-01", "2024-07-31").date if CAL.is_session(d)]
    frame = pd.DataFrame({"Date": pd.to_datetime(sessions), "Close": np.linspace(100, 110, len(sessions))})
    items = [classified(i, ny(2024, 7, 1) + timedelta(hours=9 * i), [EARN, LEGAL, ANALYST, NOISE][i % 4])
             for i in range(60)]
    out = {h: {r["trading_date"]: r for r in generate_event_features(items, build_targets(frame, horizon=h)["Date"],
                                                                       CAL, ["AAPL"])} for h in pair}
    shared = set(out[1]) & set(out[5])
    assert shared and all(out[1][d] == out[5][d] for d in shared)


# ==========================================
# Symbols, duplicates, determinism, schema
# ==========================================


def test_symbol_separation_and_multi_symbol_articles():
    items = [classified(1, ny(2024, 7, 10, 9, 0), EARN, symbols=("AAPL",)),
             classified(2, ny(2024, 7, 10, 10, 0), LEGAL, symbols=("MSFT",)),
             classified(3, ny(2024, 7, 10, 11, 0), ANALYST, symbols=("AAPL", "MSFT"))]
    r = rows_by_date(items, [D], symbols=["msft", "AAPL"])
    assert (r[("AAPL", D)]["event_count"], r[("AAPL", D)]["legal_event_count"]) == (2, 0)
    assert (r[("MSFT", D)]["event_count"], r[("MSFT", D)]["earnings_event_count"]) == (2, 0)
    assert row(items, D, "GOOG")["article_count"] == 0


def test_duplicates_do_not_inflate_and_conflicts_raise():
    a = classified(1, ny(2024, 7, 10, 10, 0), EARN)
    assert row([a, a, classified(1, ny(2024, 7, 10, 10, 0), EARN)], D)["event_count"] == 1
    with pytest.raises(EventFeatureError, match="conflicting"):
        row([a, classified(1, ny(2024, 7, 10, 10, 0), LEGAL)], D)


def test_shuffled_input_is_deterministic():
    items = [classified(i, ny(2024, 7, 8) + timedelta(hours=7 * i), [EARN, LEGAL, ANALYST, NOISE][i % 4],
                        symbols=[("AAPL",), ("AAPL", "MSFT")][i % 2]) for i in range(24)]
    dates = [date(2024, 7, d) for d in (8, 9, 10, 11, 12, 15)]
    reference = generate_event_features(items, dates, CAL, ["AAPL", "MSFT"])
    rng = random.Random(2)
    for _ in range(5):
        shuffled, ds = items[:], dates[:]
        rng.shuffle(shuffled)
        rng.shuffle(ds)
        assert generate_event_features(shuffled, ds, CAL, ["MSFT", "AAPL"]) == reference


def test_output_schema_and_input_validation():
    r = row([classified(1, ny(2024, 7, 10, 10, 0), EARN)], D)
    assert tuple(r) == EVENT_OUTPUT_COLUMNS
    assert r["event_feature_version"] == "event_features_v1" and r["trading_date"] == D
    with pytest.raises(EventFeatureError):
        generate_event_features([], [D, D], CAL, ["AAPL"])
    with pytest.raises(EventFeatureError):
        generate_event_features([], [D], CAL, "AAPL")
    with pytest.raises(EventFeatureError):
        generate_event_features(["not classified"], [D], CAL, ["AAPL"])


def test_previous_session_before():
    assert CAL.previous_session_before(date(2024, 7, 5)) == date(2024, 7, 3)            # skips July 4
    assert CAL.previous_session_before(date(2024, 7, 15)) == date(2024, 7, 12)          # skips weekend
    assert CAL.previous_session_before(date(2024, 7, 10), 5) == date(2024, 7, 2)
    with pytest.raises(ValueError):
        CAL.previous_session_before(date(2024, 7, 10), 0)
