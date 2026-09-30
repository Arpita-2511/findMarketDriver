"""
Phase 4C orchestration, end to end with the REAL AlpacaNewsProvider over
mocked HTTP: ingestion run -> chunk snapshots -> manifest -> canonical dataset.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from services.historical_news_ingestion import (
    IngestionConfig,
    chunk_intervals,
    create_provider,
    run_historical_ingestion,
)
from services.news_dataset import build_canonical_dataset, load_canonical_dataset
from services.news_provider import NewsAuthError, NewsPaginationError, NewsProviderError
from services.news_schema import NewsValidationError
from tests.news_fakes import KEY, SECRET, FakeResponse, FakeSession, make_provider, page, raw

UTC = timezone.utc
START = datetime(2017, 1, 1, tzinfo=UTC)
END = datetime(2017, 1, 3, tzinfo=UTC)
RUN_CLOCK = datetime(2026, 10, 1, 13, 0, tzinfo=UTC)


def config(**overrides) -> IngestionConfig:
    base = dict(provider="alpaca", symbols=("AAPL",), start=START, end=END, chunk_days=365)
    base.update(overrides)
    return IngestionConfig(**base)


def run(tmp_path, *responses, **overrides):
    """One ingestion run + dataset build into tmp_path. Returns (session, manifest_path, dataset_path, meta)."""
    session = FakeSession(*responses)
    manifest = run_historical_ingestion(make_provider(session), config(**overrides),
                                        raw_dir=tmp_path / "raw", clock=lambda: RUN_CLOCK)
    dataset, meta = build_canonical_dataset(manifest, tmp_path / "processed")
    return session, manifest, dataset, meta


def dataset_ids(path):
    articles, _ = load_canonical_dataset(path)
    return [a.provider_article_id for a in articles]


def no_manifest(tmp_path) -> bool:
    return not list((tmp_path / "raw").glob("*.manifest.json")) and not (tmp_path / "processed").exists()


# ==========================================
# Pagination (through the existing provider)
# ==========================================


def test_single_page_ingestion(tmp_path):
    session, manifest, dataset, meta = run(tmp_path, page([raw(1), raw(2)]))
    assert len(session.calls) == 1
    assert dataset_ids(dataset) == ["1", "2"]
    m = json.loads(manifest.read_text(encoding="utf-8"))
    assert (m["totals"]["chunks"], m["totals"]["pages"], meta["rows"]) == (1, 1, 2)


def test_multi_page_ingestion_runs_until_token_absent(tmp_path):
    session, manifest, dataset, meta = run(
        tmp_path, page([raw(1), raw(2)], "tok1"), page([raw(3)], "tok2"), page([raw(4)], None), page_size=2)
    assert [c["params"].get("page_token") for c in session.calls] == [None, "tok1", "tok2"]
    assert dataset_ids(dataset) == ["1", "2", "3", "4"]
    assert json.loads(manifest.read_text(encoding="utf-8"))["totals"]["pages"] == 3


def test_duplicate_ids_within_response_and_across_pages(tmp_path):
    _, manifest, dataset, meta = run(tmp_path, page([raw(1), raw(1), raw(2)], "t"), page([raw(2), raw(3)]))
    assert dataset_ids(dataset) == ["1", "2", "3"]
    chunk = json.loads(manifest.read_text(encoding="utf-8"))["chunks"][0]
    assert chunk["n_raw_articles"] == 5 and chunk["n_duplicates_removed"] == 2


def test_duplicates_across_chunks_keep_latest_version(tmp_path):
    boundary = "2017-01-02T00:00:00Z"                                # end of chunk 1 == start of chunk 2
    old = raw(5, created=boundary)
    revised = raw(5, created=boundary, updated="2017-01-02T06:00:00Z", headline="Revised")
    _, _, dataset, meta = run(tmp_path, page([raw(1, created="2017-01-01T10:00:00Z"), old]),
                              page([revised, raw(6)]), chunk_days=1)
    articles, _ = load_canonical_dataset(dataset)
    five = [a for a in articles if a.provider_article_id == "5"]
    assert len(five) == 1 and five[0].headline == "Revised"
    assert five[0].information_available_at == datetime(2017, 1, 2, 6, 0, tzinfo=UTC)
    assert (meta["n_duplicates_removed"], meta["n_conflicting_duplicates"]) == (1, 1)


def test_pagination_failure_propagates_without_manifest(tmp_path):
    with pytest.raises(NewsPaginationError):
        run(tmp_path, page([raw(1)], "same"), page([raw(2)], "same"))
    assert no_manifest(tmp_path)


# ==========================================
# Deterministic ordering
# ==========================================


def test_output_does_not_depend_on_provider_sort_order(tmp_path):
    items = [raw(1, created="2017-01-01T09:00:00Z"), raw(2, created="2017-01-02T09:00:00Z"),
             raw(3, created="2017-01-01T20:00:00Z")]
    _, _, asc, _ = run(tmp_path / "a", page(items))
    _, _, desc, _ = run(tmp_path / "b", page(list(reversed(items))), sort="desc")
    assert dataset_ids(asc) == ["1", "3", "2"]                       # by created_at
    assert asc.read_bytes() == desc.read_bytes()


# ==========================================
# Symbol filtering
# ==========================================


def test_symbol_filtering_and_multi_symbol_articles(tmp_path):
    items = [raw(1), raw(2, symbols=("GOOG",)), raw(3, symbols=("MSFT", "AAPL"))]
    _, _, dataset, meta = run(tmp_path, page(items))
    articles, _ = load_canonical_dataset(dataset)
    assert [a.provider_article_id for a in articles] == ["1", "3"]
    assert articles[1].symbols == ("AAPL", "MSFT")                   # kept intact, nothing added/removed
    assert meta["n_without_requested_symbol"] == 1


def test_multiple_requested_symbols(tmp_path):
    items = [raw(1), raw(2, symbols=("MSFT",)), raw(3, symbols=("GOOG",))]
    session, _, dataset, _ = run(tmp_path, page(items), symbols=("msft", "AAPL"))
    assert session.calls[0]["params"]["symbols"] == "AAPL,MSFT"
    assert dataset_ids(dataset) == ["1", "2"]


# ==========================================
# Interval boundaries: start <= created_at < end
# ==========================================


def test_start_boundary(tmp_path):
    items = [raw(1, created="2016-12-31T23:59:59Z"), raw(2, created="2017-01-01T00:00:00Z")]
    _, _, dataset, meta = run(tmp_path, page(items))
    assert dataset_ids(dataset) == ["2"] and meta["n_outside_interval"] == 1


def test_end_boundary_and_updates_after_end(tmp_path):
    items = [raw(1, created="2017-01-02T23:59:59Z"), raw(2, created="2017-01-03T00:00:00Z"),
             raw(3, created="2017-01-02T12:00:00Z", updated="2017-01-05T08:00:00Z")]
    _, _, dataset, meta = run(tmp_path, page(items))
    articles, _ = load_canonical_dataset(dataset)
    assert [a.provider_article_id for a in articles] == ["3", "1"]
    late = articles[0]
    assert late.created_at == datetime(2017, 1, 2, 12, 0, tzinfo=UTC)              # unchanged
    assert late.information_available_at == datetime(2017, 1, 5, 8, 0, tzinfo=UTC)  # unchanged, after END
    assert meta["n_outside_interval"] == 1


def test_chunk_intervals_tile_exactly():
    s, e = datetime(2017, 1, 1, tzinfo=UTC), datetime(2017, 3, 15, 12, tzinfo=UTC)
    chunks = chunk_intervals(s, e, 30)
    assert chunks[0][0] == s and chunks[-1][1] == e
    assert all(a1 == b0 for (_, b0), (a1, _) in zip(chunks, chunks[1:]))            # contiguous, no overlap
    assert all(b - a <= timedelta(days=30) for a, b in chunks)
    assert len(chunks) == 3


# ==========================================
# Empty / malformed / invalid / provider failures
# ==========================================


def test_empty_response(tmp_path):
    _, _, dataset, meta = run(tmp_path, page([]))
    assert dataset.read_bytes() == b"" and meta["rows"] == 0 and meta["first_created_at"] is None


def test_malformed_record_fails_run(tmp_path):
    with pytest.raises(NewsValidationError, match="headline"):
        run(tmp_path, page([raw(1), raw(2, headline=None)]))
    assert no_manifest(tmp_path)


@pytest.mark.parametrize("bad", ["not-a-date", "2017-01-02T10:00:00"])   # garbage, naive
def test_invalid_timestamp_fails_run(tmp_path, bad):
    with pytest.raises(NewsValidationError):
        run(tmp_path, page([raw(1, created=bad)]))
    assert no_manifest(tmp_path)


def test_provider_error_propagates(tmp_path):
    with pytest.raises(NewsProviderError, match="after 4 attempts"):
        run(tmp_path, *[FakeResponse(500) for _ in range(4)])
    assert no_manifest(tmp_path)


def test_failure_in_later_chunk_leaves_no_manifest(tmp_path):
    with pytest.raises(NewsProviderError):
        run(tmp_path, page([raw(1, created="2017-01-01T10:00:00Z")]),
            *[FakeResponse(503) for _ in range(4)], chunk_days=1)
    assert len(list((tmp_path / "raw").glob("*.jsonl"))) == 1                     # first chunk only
    assert no_manifest(tmp_path)


# ==========================================
# Credentials
# ==========================================


def test_auth_failure_does_not_leak_credentials(tmp_path):
    with pytest.raises(NewsAuthError) as exc:
        run(tmp_path, FakeResponse(401, text=f"forbidden {KEY} {SECRET}"))
    assert KEY not in str(exc.value) and SECRET not in str(exc.value)
    assert no_manifest(tmp_path)


def test_no_credentials_in_any_written_file(tmp_path):
    session, _, _, _ = run(tmp_path, page([raw(1)], "t"), page([raw(2)]))
    assert session.calls[0]["headers"]["APCA-API-SECRET-KEY"] == SECRET      # sent as a header only
    for path in tmp_path.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert KEY not in text and SECRET not in text, path.name


# ==========================================
# Provenance & configuration
# ==========================================


def test_provenance_metadata(tmp_path):
    _, manifest_path, dataset, meta = run(tmp_path, page([raw(1), raw(1)]), include_content=False, page_size=10)
    m = json.loads(manifest_path.read_text(encoding="utf-8"))

    for key in ["run_id", "provider", "symbols", "start", "end", "chunk_days", "include_content",
                "page_size", "interval_rule", "started_at", "finished_at", "chunks", "totals"]:
        assert key in m
    assert m["include_content"] is False and m["page_size"] == 10
    assert m["start"] == START.isoformat() and m["end"] == END.isoformat()
    chunk = m["chunks"][0]
    assert chunk["pages"] == 1 and chunk["n_duplicates_removed"] == 1
    from training.build_dataset import file_sha256
    assert chunk["sha256"] == file_sha256(manifest_path.parent / chunk["snapshot"])

    assert meta["manifest"]["sha256"] == file_sha256(manifest_path)
    assert meta["source_snapshots"] == [{"file": chunk["snapshot"], "sha256": chunk["sha256"]}]
    assert meta["sha256"] == file_sha256(dataset)
    assert meta["include_content"] is False
    for key in ["interval_rule", "symbol_rule", "dedup_rule", "availability_rule", "schema_version"]:
        assert meta[key]


@pytest.mark.parametrize("overrides, message", [
    ({"symbols": ("AA PL",)}, "invalid ticker"),
    ({"symbols": ("",)}, "invalid ticker"),
    ({"symbols": "AAPL"}, "non-empty sequence"),
    ({"start": datetime(2017, 1, 1)}, "timezone-aware"),
    ({"start": END}, "before end"),
    ({"chunk_days": 0}, "chunk_days"),
    ({"sort": "random"}, "sort"),
])
def test_invalid_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        config(**overrides)


def test_provider_mismatch_and_unknown_provider(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        run_historical_ingestion(make_provider(FakeSession()), config(provider="newsdata"), raw_dir=tmp_path)
    with pytest.raises(ValueError, match="unknown news provider"):
        create_provider("nope")
