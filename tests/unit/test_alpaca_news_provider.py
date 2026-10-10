"""AlpacaNewsProvider with mocked HTTP - no network, no real credentials."""

import json
from datetime import datetime, timedelta, timezone

import pytest
import requests

from services.alpaca_news_provider import (
    ALPACA_MAX_PAGE_SIZE,
    API_KEY_ENV,
    API_SECRET_ENV,
    MALFORMED_POLICY_AMENDMENT_SHA256,
    MALFORMED_REASONS,
    AlpacaNewsProvider,
    alpaca_to_canonical,
    chunk_malformed_threshold,
    classify_malformed,
)
from services.news_provider import NewsAuthError, NewsPaginationError, NewsProviderError, NewsQuery
from services.news_schema import ML_FACING_FIELDS, NewsArticle, NewsValidationError

UTC = timezone.utc
KEY = "TEST-KEY-ID-not-real"
SECRET = "TEST-SECRET-not-real-9f8e7d"
FETCHED = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, text="", bad_json=False):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = text
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params), "headers": dict(headers), "timeout": timeout})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def raw(article_id, created="2017-01-03T14:30:00Z", updated=None, symbols=("AAPL",), **extra):
    item = {"id": article_id, "headline": f"Headline {article_id}", "author": "Benzinga Newsdesk",
            "created_at": created, "updated_at": updated or created, "summary": "s",
            "content": "<p>body</p>", "url": f"https://www.benzinga.com/{article_id}",
            "images": [{"size": "thumb", "url": "https://img"}], "symbols": list(symbols),
            "source": "benzinga"}
    item.update(extra)
    return item


def page(items, token=None):
    return FakeResponse(payload={"news": items, "next_page_token": token})


def provider(session, **kwargs):
    sleeps = kwargs.pop("sleeps", [])
    return AlpacaNewsProvider(KEY, SECRET, session=session, sleep=sleeps.append,
                              clock=lambda: FETCHED, **kwargs)


def query(**kwargs):
    base = dict(symbols=("AAPL",), start=datetime(2017, 1, 1, tzinfo=UTC), end=datetime(2017, 2, 1, tzinfo=UTC))
    base.update(kwargs)
    return NewsQuery(**base)


def assert_no_secret(*texts):
    for text in texts:
        assert KEY not in text and SECRET not in text


# ==========================================
# 1. Response -> canonical schema
# ==========================================


def test_conversion_to_canonical():
    a = alpaca_to_canonical(raw(24803233, updated="2017-01-03T14:35:00Z"), FETCHED)
    assert a.provider == "alpaca" and a.provider_article_id == "24803233"
    assert a.headline == "Headline 24803233"
    assert a.symbols == ("AAPL",)
    assert a.source == "benzinga"
    assert a.source_url == "https://www.benzinga.com/24803233"
    assert a.content == "<p>body</p>" and a.summary == "s"
    assert a.created_at == datetime(2017, 1, 3, 14, 30, tzinfo=UTC)
    assert a.fetched_at == FETCHED
    assert a.provider_metadata["author"] == "Benzinga Newsdesk"
    assert a.provider_metadata["images"] == [{"size": "thumb", "url": "https://img"}]
    assert a.provider_metadata["raw_created_at"] == "2017-01-03T14:30:00Z"


# ==========================================
# 2. Required fields
# ==========================================


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r.pop("id"), "no id"),
    (lambda r: r.update(headline=None), "headline"),
    (lambda r: r.pop("created_at"), "created_at"),
    (lambda r: r.update(symbols="AAPL"), "symbols"),
])
def test_required_field_validation(mutate, message):
    item = raw(1)
    mutate(item)
    with pytest.raises(NewsValidationError, match=message):
        alpaca_to_canonical(item, FETCHED)


# Amendment 04 (Rev 3, ADOPTED): malformed records are excluded, counted and
# capped; this replaces the superseded single-record abort test.

def _without(key):
    return lambda r: r.pop(key)


def _set(key, value):
    return lambda r: r.update({key: value})


MALFORMED_CASES = [
    (_without("id"), "missing_or_blank_id"),
    (_set("id", None), "missing_or_blank_id"),
    (_set("id", "   "), "missing_or_blank_id"),
    (_without("headline"), "missing_or_blank_headline"),
    (_set("headline", None), "missing_or_blank_headline"),
    (_set("headline", ""), "missing_or_blank_headline"),
    (_set("headline", " \t "), "missing_or_blank_headline"),
    (_without("created_at"), "missing_created_at"),
    (_set("created_at", None), "missing_created_at"),
    (_set("created_at", "not-a-date"), "missing_created_at"),
    (_set("created_at", "2017-01-03T14:30:00"), "missing_created_at"),        # naive
    (_set("updated_at", "garbage"), "invalid_updated_at"),
    (_set("updated_at", "2017-01-03T14:35:00"), "invalid_updated_at"),        # naive
    (_set("symbols", "AAPL"), "invalid_symbols_type"),
    (_set("symbols", ["AAPL", 5]), "invalid_symbols_type"),
    (_set("symbols", ["AAPL", " "]), "invalid_symbols_type"),
]


@pytest.mark.parametrize("mutate, reason", MALFORMED_CASES)
def test_malformed_record_is_excluded_counted_and_never_repaired(mutate, reason):
    bad = raw(2)
    mutate(bad)
    assert classify_malformed(bad) == reason
    with pytest.raises(NewsValidationError):                     # the strict converter still rejects it
        alpaca_to_canonical(bad, FETCHED)

    result = provider(FakeSession(page([raw(1), bad, raw(3)]))).fetch_historical(query())
    p = result.provenance
    assert [a.provider_article_id for a in result.articles] == ["1", "3"]
    assert (p["n_raw_records_fetched"], p["n_malformed_excluded"], p["n_raw_articles"]) == (3, 1, 2)
    assert p["malformed_reason_counts"] == {r: int(r == reason) for r in MALFORMED_REASONS}
    assert p["malformed_excluded"] == [{
        "provider_article_id": "<missing>" if reason == "missing_or_blank_id" else "2", "reason": reason,
        "chunk_start_utc": "2017-01-01T00:00:00Z", "chunk_end_utc": "2017-02-01T00:00:00Z"}]
    assert p["malformed_fraction"] == 1 / 3
    assert p["malformed_policy"]["amendment_sha256"] == MALFORMED_POLICY_AMENDMENT_SHA256


def test_classification_order_is_deterministic():
    item = raw(1, created="nope")
    item.pop("id")
    item["headline"] = None
    assert classify_malformed(item) == "missing_or_blank_id"
    item["id"] = 7
    assert classify_malformed(item) == "missing_or_blank_headline"


@pytest.mark.parametrize("mutate", [_set("headline", 123), _set("summary", 5), _set("source", ["x"])])
def test_violation_outside_the_categories_fails_the_chunk(mutate):
    bad = raw(2)
    mutate(bad)
    assert classify_malformed(bad) is None
    with pytest.raises(NewsValidationError):
        provider(FakeSession(page([raw(1), bad]))).fetch_historical(query())


def test_non_object_item_fails_the_chunk():
    with pytest.raises(NewsValidationError, match="not an object"):
        provider(FakeSession(page([raw(1), "junk"]))).fetch_historical(query())


def test_missing_or_empty_symbols_and_falsy_updated_at_stay_valid():
    for item in (raw(1, symbols=()), {**raw(2), "updated_at": None}, {k: v for k, v in raw(3).items() if k != "symbols"}):
        assert classify_malformed(item) is None
        alpaca_to_canonical(item, FETCHED)


def test_updated_before_created_is_kept_not_excluded():
    item = raw(1, created="2017-01-03T14:30:00Z", updated="2017-01-03T14:00:00Z")
    assert classify_malformed(item) is None
    result = provider(FakeSession(page([item]))).fetch_historical(query())
    assert len(result.articles) == 1 and result.provenance["n_malformed_excluded"] == 0


def test_schema_still_rejects_blank_or_missing_headline_directly():
    base = dict(provider="alpaca", provider_article_id="1", symbols=("AAPL",), created_at=FETCHED,
                information_available_at=FETCHED, fetched_at=FETCHED)
    for headline in ("", "   ", None):
        with pytest.raises(NewsValidationError, match="headline is required"):
            NewsArticle(headline=headline, **base)


def test_raw_count_is_measured_before_conversion_and_dedup():
    bad = raw(9, headline=None)
    session = FakeSession(page([raw(1), raw(2), bad], "t"), page([raw(2), raw(3)]))
    p = provider(session).fetch_historical(query()).provenance
    assert p["n_raw_records_fetched"] == 5                 # every record of every page
    assert p["n_malformed_excluded"] == 1
    assert p["n_raw_articles"] == 4                         # converted, before dedup
    assert (p["n_duplicates_removed"], p["n_articles"]) == (1, 3)


def _chunk_page(n_valid, n_bad):
    return page([raw(i) for i in range(n_valid)] + [raw(100_000 + i, headline=None) for i in range(n_bad)])


@pytest.mark.parametrize("n_raw, n_bad, passes", [
    (500, 10, True), (500, 11, False),              # ABS_FLOOR binds: max(10, 5.0) = 10
    (1000, 10, True), (1000, 11, False),            # tie: max(10, 10.0) = 10
    (2345, 23, True), (2345, 24, False),            # fractional: max(10, 23.45) = 23.45, no rounding
])
def test_per_chunk_cap_boundaries(n_raw, n_bad, passes):
    session = FakeSession(_chunk_page(n_raw - n_bad, n_bad))
    if passes:
        p = provider(session).fetch_historical(query()).provenance
        assert (p["n_raw_records_fetched"], p["n_malformed_excluded"]) == (n_raw, n_bad)
    else:
        with pytest.raises(NewsProviderError, match="per-chunk cap exceeded") as exc:
            provider(session).fetch_historical(query())
        assert_no_secret(str(exc.value))


def test_chunk_threshold_is_not_rounded():
    assert chunk_malformed_threshold(2345) == 23.45
    assert chunk_malformed_threshold(0) == 10 and chunk_malformed_threshold(500) == 10


# ==========================================
# 3. Timezones
# ==========================================


def test_offset_timestamps_normalized_to_utc():
    a = alpaca_to_canonical(raw(1, created="2017-01-03T09:30:00-05:00"), FETCHED)
    assert a.created_at == datetime(2017, 1, 3, 14, 30, tzinfo=UTC)


def test_naive_provider_timestamp_rejected():
    with pytest.raises(NewsValidationError, match="timezone-aware"):
        alpaca_to_canonical(raw(1, created="2017-01-03T14:30:00"), FETCHED)


def test_query_requires_aware_datetimes():
    with pytest.raises(ValueError, match="timezone-aware"):
        query(start=datetime(2017, 1, 1))


# ==========================================
# 4-5. Pagination and page-token progression
# ==========================================


def test_pagination_follows_tokens_and_keeps_params_fixed():
    session = FakeSession(page([raw(1), raw(2)], "tok1"), page([raw(3)], "tok2"), page([raw(4)], None))
    result = provider(session).fetch_historical(query(page_size=2))

    assert [a.provider_article_id for a in result.articles] == ["1", "2", "3", "4"]
    assert [c["params"].get("page_token") for c in session.calls] == [None, "tok1", "tok2"]
    for call in session.calls:
        p = call["params"]
        assert (p["symbols"], p["start"], p["end"], p["limit"], p["sort"], p["include_content"]) == (
            "AAPL", "2017-01-01T00:00:00Z", "2017-02-01T00:00:00Z", 2, "asc", "true")
    assert result.provenance["pages"] == 3


def test_default_page_size_is_alpaca_maximum():
    session = FakeSession(page([raw(1)]))
    provider(session).fetch_historical(query())
    assert session.calls[0]["params"]["limit"] == ALPACA_MAX_PAGE_SIZE == 50


def test_page_size_above_limit_rejected_before_any_request():
    session = FakeSession()
    with pytest.raises(ValueError, match="between 1 and 50"):
        provider(session).fetch_historical(query(page_size=51))
    assert session.calls == []


def test_descending_sort_is_passed_through():
    session = FakeSession(page([raw(2), raw(1)]))
    provider(session).fetch_historical(query(sort="desc"))
    assert session.calls[0]["params"]["sort"] == "desc"


def test_repeated_page_token_stops_instead_of_looping():
    session = FakeSession(page([raw(1)], "same"), page([raw(2)], "same"))
    with pytest.raises(NewsPaginationError, match="repeated page token"):
        provider(session).fetch_historical(query())


def test_page_cap_stops_with_error_not_silent_truncation():
    session = FakeSession(page([raw(1)], "a"), page([raw(2)], "b"))
    with pytest.raises(NewsPaginationError, match="max_pages=2"):
        provider(session).fetch_historical(query(), max_pages=2)
    assert len(session.calls) == 2


def test_empty_result():
    result = provider(FakeSession(page([]))).fetch_historical(query())
    assert result.articles == [] and result.provenance["n_articles"] == 0


# ==========================================
# 6. Duplicates
# ==========================================


def test_duplicates_across_pages_removed_and_counted():
    changed = raw(2)
    changed["headline"] = "Edited headline"
    session = FakeSession(page([raw(1), raw(2)], "t"), page([raw(2), changed, raw(3)]))
    result = provider(session).fetch_historical(query())

    assert [a.provider_article_id for a in result.articles] == ["1", "2", "3"]
    assert result.articles[1].headline == "Headline 2"          # first occurrence kept
    assert result.provenance["n_raw_articles"] == 5
    assert result.provenance["n_duplicates_removed"] == 2
    assert result.provenance["n_conflicting_duplicates"] == 1


# ==========================================
# 7. Multi-symbol articles
# ==========================================


def test_multi_symbol_articles_and_queries():
    session = FakeSession(page([raw(1, symbols=("msft", "AAPL", "aapl")), raw(2, symbols=("GOOG",))]))
    result = provider(session).fetch_historical(query(symbols=("aapl", "MSFT")))
    assert session.calls[0]["params"]["symbols"] == "AAPL,MSFT"
    assert result.articles[0].symbols == ("AAPL", "MSFT")
    assert result.provenance["n_without_requested_symbol"] == 1   # GOOG-only article flagged, kept


# ==========================================
# 8. Missing optional content
# ==========================================


def test_missing_optional_fields_become_none():
    item = raw(1, content="", summary=None)
    item.pop("url")
    item.pop("source")
    a = alpaca_to_canonical(item, FETCHED)
    assert (a.content, a.summary, a.source_url, a.source) == (None, None, None, None)


def test_include_content_false_is_sent():
    session = FakeSession(page([raw(1, content=None)]))
    provider(session).fetch_historical(query(include_content=False))
    assert session.calls[0]["params"]["include_content"] == "false"


# ==========================================
# 9. API errors
# ==========================================


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_are_not_retried(status):
    session = FakeSession(FakeResponse(status, text=f"forbidden {SECRET}"))
    with pytest.raises(NewsAuthError, match=str(status)) as exc:
        provider(session).fetch_historical(query())
    assert len(session.calls) == 1
    assert_no_secret(str(exc.value))


def test_client_error_not_retried_and_body_redacted():
    session = FakeSession(FakeResponse(422, text=f'{{"message":"invalid start"}} key={KEY}'))
    with pytest.raises(NewsProviderError, match="HTTP 422") as exc:
        provider(session).fetch_historical(query())
    assert len(session.calls) == 1
    assert_no_secret(str(exc.value))


def test_server_error_retried_then_succeeds():
    sleeps = []
    session = FakeSession(FakeResponse(500), FakeResponse(503), page([raw(1)]))
    result = provider(session, sleeps=sleeps, backoff_seconds=1.0).fetch_historical(query())
    assert len(result.articles) == 1
    assert sleeps == [1.0, 2.0]                                   # exponential backoff


def test_rate_limit_honours_retry_after():
    sleeps = []
    session = FakeSession(FakeResponse(429, headers={"Retry-After": "7"}), page([raw(1)]))
    provider(session, sleeps=sleeps).fetch_historical(query())
    assert sleeps == [7.0]


def test_retries_are_bounded():
    session = FakeSession(*[FakeResponse(500) for _ in range(4)])
    with pytest.raises(NewsProviderError, match="after 4 attempts"):
        provider(session, max_retries=3).fetch_historical(query())
    assert len(session.calls) == 4


@pytest.mark.parametrize("response, message", [
    (FakeResponse(200, bad_json=True), "non-JSON"),
    (FakeResponse(200, payload=[1, 2]), "unexpected JSON"),
    (FakeResponse(200, payload={"items": []}), "no 'news' list"),
])
def test_malformed_responses(response, message):
    with pytest.raises(NewsProviderError, match=message):
        provider(FakeSession(response)).fetch_historical(query())


# ==========================================
# 10. Timeouts / network failures
# ==========================================


def test_timeout_retried_then_succeeds():
    session = FakeSession(requests.Timeout("read timed out"), page([raw(1)]))
    result = provider(session).fetch_historical(query())
    assert len(result.articles) == 1
    assert session.calls[0]["timeout"] == (5.0, 30.0)


def test_persistent_connection_failure_raises_provider_error():
    session = FakeSession(*[requests.ConnectionError("unreachable") for _ in range(3)])
    with pytest.raises(NewsProviderError, match="ConnectionError") as exc:
        provider(session, max_retries=2).fetch_historical(query())
    assert_no_secret(str(exc.value))


# ==========================================
# 11. Credentials
# ==========================================


def test_missing_credentials_error_names_variables_only(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv(API_SECRET_ENV, raising=False)
    with pytest.raises(NewsAuthError, match="ALPACA_API_KEY and ALPACA_API_SECRET"):
        AlpacaNewsProvider.from_env()


def test_partial_credentials_rejected(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, KEY)
    monkeypatch.delenv(API_SECRET_ENV, raising=False)
    with pytest.raises(NewsAuthError) as exc:
        AlpacaNewsProvider.from_env()
    assert_no_secret(str(exc.value))


def test_invalid_header_error_does_not_leak_secret_into_traceback():
    """requests.InvalidHeader embeds the header VALUE in its message (e.g. a secret with a stray newline)."""
    import traceback

    leaking = requests.exceptions.InvalidHeader(
        f"Invalid leading whitespace, reserved character(s), or return character(s) in header value: {SECRET!r}")
    session = FakeSession(leaking)
    with pytest.raises(NewsProviderError, match="InvalidHeader") as exc:
        provider(session).fetch_historical(query())

    assert len(session.calls) == 1                     # not retried
    assert exc.value.__cause__ is None and exc.value.__suppress_context__
    assert_no_secret(str(exc.value), "".join(traceback.format_exception(exc.value)))


def test_credentials_only_in_headers_never_in_repr_params_or_provenance():
    session = FakeSession(page([raw(1)]))
    p = provider(session)
    result = p.fetch_historical(query())

    headers = session.calls[0]["headers"]
    assert headers["APCA-API-KEY-ID"] == KEY and headers["APCA-API-SECRET-KEY"] == SECRET
    assert_no_secret(repr(p), json.dumps(session.calls[0]["params"]),
                     json.dumps(result.provenance, default=str),
                     json.dumps([a.to_dict() for a in result.articles]))


# ==========================================
# 12. information_available_at
# ==========================================


def test_information_available_at_semantics():
    created = "2017-01-03T14:30:00Z"
    later = alpaca_to_canonical(raw(1, created=created, updated="2017-01-04T09:00:00Z"), FETCHED)
    same = alpaca_to_canonical(raw(2, created=created, updated=created), FETCHED)
    earlier = alpaca_to_canonical(raw(3, created=created, updated="2017-01-03T14:00:00Z"), FETCHED)

    assert later.information_available_at == datetime(2017, 1, 4, 9, 0, tzinfo=UTC)
    assert later.created_at == datetime(2017, 1, 3, 14, 30, tzinfo=UTC)   # never replaced
    assert same.information_available_at == same.created_at
    assert earlier.information_available_at == earlier.created_at
    assert earlier.updated_at == datetime(2017, 1, 3, 14, 0, tzinfo=UTC)  # kept as reported


def test_provenance_records_rule_and_window_check():
    session = FakeSession(page([raw(1), raw(2, created="2016-12-31T23:00:00Z")]))
    result = provider(session).fetch_historical(query())
    assert result.provenance["availability_rule"].startswith("historical_backfill_v1")
    assert result.provenance["n_created_outside_window"] == 1


# ==========================================
# 13. Provider details do not reach the ML-facing schema
# ==========================================


def test_ml_view_contains_no_alpaca_details():
    view = alpaca_to_canonical(raw(1, extra_vendor_field="x"), FETCHED).ml_view()
    assert tuple(view) == ML_FACING_FIELDS
    for leaked in ("provider", "provider_metadata", "author", "images", "url",
                   "extra_vendor_field", "raw_created_at", "fetched_at"):
        assert leaked not in view
    assert timedelta(0) <= view["information_available_at"] - view["created_at"]
