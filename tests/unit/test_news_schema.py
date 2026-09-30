"""Canonical news schema (services/news_schema.py)."""

from datetime import datetime, timedelta, timezone

import pytest

from services.news_schema import (
    ML_FACING_FIELDS,
    NewsArticle,
    NewsValidationError,
    historical_information_available_at,
    parse_timestamp,
)

UTC = timezone.utc
T0 = datetime(2017, 1, 3, 14, 30, tzinfo=UTC)


def article(**overrides) -> NewsArticle:
    fields = dict(
        provider="alpaca", provider_article_id="1", headline="Apple news",
        symbols=("AAPL",), created_at=T0, information_available_at=T0,
        fetched_at=T0 + timedelta(days=3000),
    )
    fields.update(overrides)
    return NewsArticle(**fields)


def test_normalization():
    a = article(symbols=["msft", "AAPL", "aapl"], summary="   ", content="",
                created_at=datetime(2017, 1, 3, 9, 30, tzinfo=timezone(timedelta(hours=-5))))
    assert a.symbols == ("AAPL", "MSFT")
    assert a.summary is None and a.content is None
    assert a.created_at == T0 and a.created_at.tzinfo == UTC


@pytest.mark.parametrize("name", ["created_at", "information_available_at", "fetched_at", "updated_at"])
def test_naive_timestamps_rejected(name):
    with pytest.raises(NewsValidationError, match="timezone-aware"):
        article(**{name: datetime(2017, 1, 3, 14, 30)})


@pytest.mark.parametrize("overrides, message", [
    ({"headline": "  "}, "headline"),
    ({"provider_article_id": ""}, "provider_article_id"),
    ({"provider": ""}, "provider"),
    ({"symbols": "AAPL"}, "symbols"),
    ({"summary": 5}, "summary"),
    ({"information_available_at": T0 - timedelta(seconds=1)}, "precedes created_at"),
])
def test_required_fields_and_invariants(overrides, message):
    with pytest.raises(NewsValidationError, match=message):
        article(**overrides)


@pytest.mark.parametrize("text, expected", [
    ("2017-01-03T14:30:00Z", T0),
    ("2017-01-03T20:00:00+05:30", T0),
    ("2017-01-03T14:30:00.000Z", T0),
])
def test_parse_timestamp(text, expected):
    assert parse_timestamp(text, "t") == expected


@pytest.mark.parametrize("bad", ["2017-01-03T14:30:00", "yesterday", "", None])
def test_parse_timestamp_rejects(bad):
    with pytest.raises(NewsValidationError):
        parse_timestamp(bad, "t")


def test_availability_rule():
    later, earlier = T0 + timedelta(hours=2), T0 - timedelta(hours=1)
    assert historical_information_available_at(T0, None) == T0
    assert historical_information_available_at(T0, later) == later
    assert historical_information_available_at(T0, earlier) == T0


def test_round_trip():
    a = article(updated_at=T0 + timedelta(minutes=5), information_available_at=T0 + timedelta(minutes=5),
                content="<p>body</p>", source="benzinga", source_url="https://x",
                provider_metadata={"author": "A"})
    assert NewsArticle.from_dict(a.to_dict()) == a


def test_ml_view_has_no_provider_fields():
    view = article(provider_metadata={"author": "A", "images": []}).ml_view()
    assert tuple(view) == ML_FACING_FIELDS
    for leaked in ("provider", "provider_article_id", "provider_metadata", "fetched_at",
                   "source_url", "author", "images"):
        assert leaked not in view
