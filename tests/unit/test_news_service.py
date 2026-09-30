"""News raw snapshots and ingestion orchestration (services/news_service.py)."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from services.news_provider import NewsFetchResult, NewsProvider, NewsQuery
from services.news_schema import NewsArticle, NewsValidationError
from services.news_service import (
    ingest_historical_news,
    load_news_snapshot,
    parse_cli_datetime,
    save_news_snapshot,
)
from training.build_dataset import file_sha256, meta_path_for

UTC = timezone.utc
T0 = datetime(2017, 1, 3, 14, 30, tzinfo=UTC)
QUERY = NewsQuery(symbols=("AAPL",), start=datetime(2017, 1, 1, tzinfo=UTC), end=datetime(2017, 2, 1, tzinfo=UTC))


def article(i: int, minutes: int) -> NewsArticle:
    created = T0 + timedelta(minutes=minutes)
    return NewsArticle(provider="alpaca", provider_article_id=str(i), headline=f"H{i}", symbols=("AAPL",),
                       created_at=created, information_available_at=created,
                       fetched_at=datetime(2026, 10, 1, tzinfo=UTC), content="<p>ü body</p>",
                       provider_metadata={"author": "A"})


def result(articles) -> NewsFetchResult:
    return NewsFetchResult(articles=list(articles), provenance={
        "provider": "alpaca", "endpoint": "https://example", "request": QUERY.describe(),
        "fetch_started_at": "2026-10-01T00:00:00+00:00", "fetch_finished_at": "2026-10-01T00:05:00+00:00",
        "pages": 1, "n_raw_articles": len(articles), "n_articles": len(articles),
        "n_duplicates_removed": 0, "n_conflicting_duplicates": 0,
        "n_created_outside_window": 0, "n_without_requested_symbol": 0,
    })


ARTICLES = [article(1, 0), article(2, 5), article(3, 10)]


def test_snapshot_round_trip_and_metadata(tmp_path):
    path = save_news_snapshot(result(ARTICLES), QUERY, tmp_path)
    assert path.name == "alpaca_AAPL_20170101_20170201_20261001T000500Z.jsonl"

    loaded, meta = load_news_snapshot(path)
    assert loaded == ARTICLES
    assert meta["sha256"] == file_sha256(path)
    assert meta["rows"] == 3 and meta["schema_version"] == "news_v1"
    assert meta["request"]["symbols"] == ["AAPL"]
    assert meta["first_created_at"] == T0.isoformat()


def test_snapshot_order_is_independent_of_request_sort(tmp_path):
    a = save_news_snapshot(result(ARTICLES), QUERY, tmp_path / "asc")
    b = save_news_snapshot(result(list(reversed(ARTICLES))), QUERY, tmp_path / "desc")
    assert a.read_bytes() == b.read_bytes()


def test_tampered_snapshot_rejected(tmp_path):
    path = save_news_snapshot(result(ARTICLES), QUERY, tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace("H2", "H9"), encoding="utf-8", newline="\n")
    with pytest.raises(NewsValidationError, match="hash mismatch"):
        load_news_snapshot(path)


def test_snapshot_without_metadata_rejected(tmp_path):
    path = save_news_snapshot(result(ARTICLES), QUERY, tmp_path)
    meta_path_for(path).unlink()
    with pytest.raises(FileNotFoundError, match="metadata"):
        load_news_snapshot(path)


def test_snapshot_never_overwritten(tmp_path):
    save_news_snapshot(result(ARTICLES), QUERY, tmp_path)
    with pytest.raises(FileExistsError):
        save_news_snapshot(result(ARTICLES), QUERY, tmp_path)


def test_empty_snapshot(tmp_path):
    loaded, meta = load_news_snapshot(save_news_snapshot(result([]), QUERY, tmp_path))
    assert loaded == [] and meta["rows"] == 0 and meta["first_created_at"] is None


class StubProvider(NewsProvider):
    name = "stub"

    def __init__(self):
        self.calls = []

    def fetch_historical(self, query, *, max_pages=None):
        self.calls.append((query, max_pages))
        return result(ARTICLES)


def test_ingest_uses_provider_interface(tmp_path):
    stub = StubProvider()
    path = ingest_historical_news(stub, QUERY, raw_dir=tmp_path, max_pages=7)
    assert stub.calls == [(QUERY, 7)]
    assert json.loads(meta_path_for(path).read_text(encoding="utf-8"))["rows"] == 3


def test_cli_dates():
    assert parse_cli_datetime("2017-01-03") == datetime(2017, 1, 3, tzinfo=UTC)
    assert parse_cli_datetime("2017-01-03T09:30:00-05:00") == T0
    with pytest.raises(Exception, match="timezone offset"):
        parse_cli_datetime("2017-01-03T14:30:00")
