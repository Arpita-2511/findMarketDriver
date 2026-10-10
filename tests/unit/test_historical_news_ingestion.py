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
from services.alpaca_news_provider import alpaca_to_canonical, malformed_policy, run_cap_policy
from services.news_dataset import build_canonical_dataset, load_canonical_dataset
from services.news_provider import NewsAuthError, NewsFetchResult, NewsPaginationError, NewsProviderError, NewsQuery
from services.news_schema import NewsValidationError
from services.news_service import load_news_snapshot, save_news_snapshot
from tests.news_fakes import FETCHED, KEY, SECRET, FakeResponse, FakeSession, make_provider, page, raw

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


# ---- Amendment 04 (Rev 3, ADOPTED): malformed-record exclusion and caps ----
# (supersedes test_malformed_record_fails_run / test_invalid_timestamp_fails_run)


def valid(n, start=1, created="2017-01-01T15:00:00Z"):
    return [raw(i, created=created) for i in range(start, start + n)]


@pytest.mark.parametrize("bad, reason", [
    (raw(9001, headline=None), "missing_or_blank_headline"),
    (raw(9001, created="not-a-date"), "missing_created_at"),
    (raw(9001, created="2017-01-02T10:00:00"), "missing_created_at"),            # naive
])
def test_malformed_record_excluded_within_caps(tmp_path, bad, reason):
    _, manifest, dataset, meta = run(tmp_path, page(valid(600) + [bad]))           # 1/601 = 0.166% <= 0.2%
    assert "9001" not in dataset_ids(dataset) and len(dataset_ids(dataset)) == 600
    m = json.loads(manifest.read_text(encoding="utf-8"))
    chunk = m["chunks"][0]
    assert (chunk["n_raw_records_fetched"], chunk["n_malformed_excluded"]) == (601, 1)
    assert chunk["malformed_reason_counts"][reason] == 1
    assert chunk["malformed_counts_source"] == "measured_this_run"
    assert (m["totals"]["n_raw_records_fetched"], m["totals"]["n_malformed_excluded"]) == (601, 1)
    assert m["totals"]["malformed_fraction_among_fresh_chunks"] == 1 / 601
    assert m["malformed_policy"] == malformed_policy()

    snap = manifest.parent / chunk["snapshot"]
    snap_meta = json.loads(snap.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert snap_meta["malformed_excluded"] == [{"provider_article_id": "9001", "reason": reason,
                                                "chunk_start_utc": "2017-01-01T00:00:00Z",
                                                "chunk_end_utc": "2017-01-03T00:00:00Z"}]
    assert b'"9001"' not in snap.read_bytes()                                      # exclusions stay out of JSONL


def test_run_cap_breach_after_save_leaves_snapshot_but_no_manifest(tmp_path):
    # 1/2 > 0.2% on final totals; per-chunk cap 10 ok
    with pytest.raises(NewsProviderError, match=r"cumulative run cap exceeded on final totals: 1/2 = 0\.5 > 0\.002"):
        run(tmp_path, page([raw(1), raw(2, headline=None)]))
    assert len(list((tmp_path / "raw").glob("*.jsonl"))) == 1
    assert no_manifest(tmp_path)


def test_per_chunk_cap_breach_leaves_no_snapshot_and_no_manifest(tmp_path):
    with pytest.raises(NewsProviderError, match="per-chunk cap exceeded"):
        run(tmp_path, page(valid(1) + [raw(9000 + i, headline=None) for i in range(11)]))
    assert not list((tmp_path / "raw").glob("*.jsonl"))
    assert no_manifest(tmp_path)


def test_final_whole_run_cap_breach_aborts_and_keeps_snapshots_valid(tmp_path):
    # all-fresh run: chunk 1 clean (600), chunk 2 1 valid + 2 malformed -> final 2/603 > 0.2%
    with pytest.raises(NewsProviderError, match=r"cumulative run cap exceeded on final totals: 2/603 = 0\.0033"):
        run(tmp_path, page(valid(600)), page(valid(1, 700, "2017-01-02T15:00:00Z")
                                             + [raw(9001, headline=None), raw(9002, headline=None)]), chunk_days=1)
    snaps = sorted((tmp_path / "raw").glob("*.jsonl"))
    assert len(snaps) == 2 and no_manifest(tmp_path)
    articles, meta = load_news_snapshot(snaps[0])                                   # hash-verified, reusable
    assert len(articles) == 600 and meta["n_malformed_excluded"] == 0


# ---- reused snapshots: inferred (pre-amendment) vs recorded (post-amendment) ----

CHUNK1_END = datetime(2017, 1, 2, tzinfo=UTC)


def legacy_snapshot(raw_dir, n, **prov_overrides):
    """A chunk-1 snapshot as the pre-amendment provider wrote it (no malformed-record fields)."""
    q = NewsQuery(symbols=("AAPL",), start=START, end=CHUNK1_END)
    arts = [alpaca_to_canonical(item, FETCHED) for item in valid(n)]
    prov = {"provider": "alpaca", "request": q.describe(), "fetch_finished_at": "2026-09-30T10:00:00+00:00",
            "pages": 1, "n_raw_articles": n, "n_articles": n, "n_duplicates_removed": 0,
            "n_conflicting_duplicates": 0, **prov_overrides}
    return save_news_snapshot(NewsFetchResult(articles=arts, provenance=prov), q, raw_dir)


def resume(tmp_path, *responses, session=None, **overrides):
    return run_historical_ingestion(make_provider(session or FakeSession(*responses)),
                                    config(chunk_days=1, **overrides),
                                    raw_dir=tmp_path / "raw", clock=lambda: RUN_CLOCK, resume=True)


def test_reused_legacy_chunk_contributes_inferred_zero(tmp_path):
    legacy = legacy_snapshot(tmp_path / "raw", 50)
    before = legacy.read_bytes(), legacy.with_suffix(".meta.json").read_bytes()
    m = json.loads(resume(tmp_path, page(valid(600, 100, "2017-01-02T15:00:00Z"))).read_text(encoding="utf-8"))
    c0, c1 = m["chunks"]
    assert (c0["reused"], c0["malformed_counts_source"], c0["n_raw_records_fetched"], c0["n_malformed_excluded"]) == \
        (True, "inferred_from_strict_pre_amendment_policy", 50, 0)
    assert (c1["reused"], c1["malformed_counts_source"]) == (False, "measured_this_run")
    assert (m["totals"]["n_inferred_chunks"], m["totals"]["n_fresh_chunks"]) == (1, 1)
    assert m["totals"]["n_raw_records_fetched"] == 650
    assert (legacy.read_bytes(), legacy.with_suffix(".meta.json").read_bytes()) == before   # never rewritten


def test_fresh_only_run_cap_not_diluted_by_inferred_chunks(tmp_path):
    legacy_snapshot(tmp_path / "raw", 2000)
    fresh = valid(500, 5000, "2017-01-02T15:00:00Z") + [raw(9001, headline=None), raw(9002, headline=None)]
    # cumulative 2/2502 = 0.08% passes; fresh-only 2/502 = 0.40% must abort
    with pytest.raises(NewsProviderError, match="fresh-chunk run cap exceeded on final totals: 2/502"):
        resume(tmp_path, page(fresh))
    assert not list((tmp_path / "raw").glob("*.manifest.json"))


@pytest.mark.parametrize("overrides", [{"n_raw_articles": 51}, {"n_raw_articles": None}])
def test_reused_legacy_chunk_with_missing_or_inconsistent_counts_is_refused(tmp_path, overrides):
    legacy_snapshot(tmp_path / "raw", 50, **overrides)
    with pytest.raises(NewsValidationError, match="refusing to infer"):
        resume(tmp_path, page(valid(600, 100, "2017-01-02T15:00:00Z")))
    assert not list((tmp_path / "raw").glob("*.manifest.json"))


def test_reused_snapshot_from_a_different_policy_is_refused(tmp_path):
    legacy_snapshot(tmp_path / "raw", 50, n_raw_records_fetched=50, n_malformed_excluded=0,
                    malformed_reason_counts={}, malformed_policy={"amendment_sha256": "0" * 64})
    with pytest.raises(NewsValidationError, match="different malformed-record policy"):
        resume(tmp_path, page(valid(600, 100, "2017-01-02T15:00:00Z")))


def test_reused_post_amendment_chunk_uses_recorded_counts(tmp_path):
    raw_dir = tmp_path / "raw"
    with pytest.raises(NewsProviderError):                                          # chunk 2 fails after retries
        run_historical_ingestion(make_provider(FakeSession(page(valid(600) + [raw(9001, headline=None)]),
                                                           *[FakeResponse(503)] * 4)),
                                 config(chunk_days=1), raw_dir=raw_dir, clock=lambda: RUN_CLOCK)
    m = json.loads(resume(tmp_path, page(valid(600, 1000, "2017-01-02T15:00:00Z"))).read_text(encoding="utf-8"))
    c0 = m["chunks"][0]
    assert (c0["reused"], c0["malformed_counts_source"], c0["n_raw_records_fetched"], c0["n_malformed_excluded"]) == \
        (True, "recorded_in_snapshot_meta", 601, 1)
    assert m["totals"]["n_fresh_chunks"] == 2 and m["totals"]["n_malformed_excluded_among_fresh_chunks"] == 1


# ---- Amendment 05 (ADOPTED): cumulative caps judged once, on final totals ----

END3 = datetime(2017, 1, 4, tzinfo=UTC)                    # three daily chunks: 01-01, 01-02, 01-03
AMENDMENT_04_SHA = "fa05a3ec4bc7562b362ce09cfe3cd1ae4d7060f453642d7015d210acc8e6bb5a"


def incident_chunk():
    """The observed pattern: 489 valid + 1 headline-less record = 1/490 fresh."""
    return page(valid(489, 10000, "2017-01-02T15:00:00Z") + [raw(9001, headline=None)])


def snapshot_bytes(raw_dir):
    return {p.name: p.read_bytes() for p in sorted(raw_dir.iterdir())}


def test_one_malformed_in_first_fresh_chunk_of_490_passes_on_final_totals(tmp_path):
    legacy_snapshot(tmp_path / "raw", 2000)                                        # inferred, not fresh
    assert 1 / 490 > 0.002                                                          # would abort under per-chunk timing
    manifest = resume(tmp_path, incident_chunk(), page(valid(60, 20000, "2017-01-03T15:00:00Z")), end=END3)
    m = json.loads(manifest.read_text(encoding="utf-8"))
    t = m["totals"]
    assert [c["malformed_counts_source"] for c in m["chunks"]] == [
        "inferred_from_strict_pre_amendment_policy", "measured_this_run", "measured_this_run"]
    assert (t["n_raw_records_fetched_among_fresh_chunks"], t["n_malformed_excluded_among_fresh_chunks"]) == (550, 1)
    assert t["malformed_fraction_among_fresh_chunks"] == 1 / 550 <= 0.002
    assert (t["n_raw_records_fetched"], t["n_malformed_excluded"]) == (2550, 1)
    assert t["malformed_fraction"] == 1 / 2550 <= 0.002                            # checked independently
    assert m["run_cap_policy"] == run_cap_policy() and m["malformed_policy"] == malformed_policy()
    _, meta = build_canonical_dataset(manifest, tmp_path / "processed")
    assert meta["rows"] == 2000 + 489 + 60


@pytest.mark.parametrize("n_clean, passes", [(9, False), (10, True)])
def test_final_fresh_only_cap_is_strict_and_unrounded(tmp_path, n_clean, passes):
    legacy_snapshot(tmp_path / "raw", 2000)
    responses = (incident_chunk(), page(valid(n_clean, 20000, "2017-01-03T15:00:00Z")))
    if passes:                                                                      # 1/500 == 0.002, not > 0.002
        t = json.loads(resume(tmp_path, *responses, end=END3).read_text(encoding="utf-8"))["totals"]
        assert t["malformed_fraction_among_fresh_chunks"] == 0.002
        return
    with pytest.raises(NewsProviderError,
                       match=r"fresh-chunk run cap exceeded on final totals: 1/499 = 0\.002004008"):
        resume(tmp_path, *responses, end=END3)                                      # cumulative 1/2499 passes
    assert no_manifest(tmp_path)
    snaps = sorted((tmp_path / "raw").glob("*.jsonl"))
    assert len(snaps) == 3                                                          # legacy + both fresh chunks kept
    for snap in snaps:
        load_news_snapshot(snap)                                                    # every snapshot hash-verifies


def test_final_cap_failure_preserves_existing_snapshots_byte_for_byte(tmp_path):
    raw_dir = tmp_path / "raw"
    legacy_snapshot(raw_dir, 2000)
    before = snapshot_bytes(raw_dir)
    with pytest.raises(NewsProviderError, match="on final totals"):
        resume(tmp_path, incident_chunk(), page(valid(5, 20000, "2017-01-03T15:00:00Z")), end=END3)
    after = snapshot_bytes(raw_dir)
    assert {k: after[k] for k in before} == before                                  # nothing pre-existing touched
    assert not [n for n in after if n.endswith(".manifest.json")] and not (tmp_path / "processed").exists()


def test_per_chunk_cap_still_aborts_immediately(tmp_path):
    session = FakeSession(page(valid(600)),
                          page(valid(1, 700, "2017-01-02T15:00:00Z") + [raw(9000 + i, headline=None) for i in range(11)]),
                          page(valid(600, 5000, "2017-01-03T15:00:00Z")))
    with pytest.raises(NewsProviderError, match="per-chunk cap exceeded"):
        run_historical_ingestion(make_provider(session), config(chunk_days=1, end=END3),
                                 raw_dir=tmp_path / "raw", clock=lambda: RUN_CLOCK)
    assert len(session.calls) == 2                                                  # chunk 3 never requested
    assert len(list((tmp_path / "raw").glob("*.jsonl"))) == 1                       # offending chunk not saved
    assert no_manifest(tmp_path)


def test_resume_reuses_saved_fresh_chunk_without_refetch(tmp_path):
    raw_dir = tmp_path / "raw"
    legacy_snapshot(raw_dir, 2000)
    with pytest.raises(NewsProviderError, match="after 4 attempts"):                # chunk 3 unreachable
        resume(tmp_path, incident_chunk(), *[FakeResponse(503)] * 4, end=END3)
    saved = sorted(raw_dir.glob("alpaca_AAPL_20170102_20170103_*"))
    before = {p.name: p.read_bytes() for p in saved}
    recorded = json.loads(next(p for p in saved if p.name.endswith(".meta.json")).read_text(encoding="utf-8"))
    assert recorded["malformed_policy"]["amendment_sha256"] == AMENDMENT_04_SHA      # snapshot-level hash unchanged

    session = FakeSession(page(valid(60, 20000, "2017-01-03T15:00:00Z")))
    m = json.loads(resume(tmp_path, session=session, end=END3).read_text(encoding="utf-8"))
    assert len(session.calls) == 1 and session.calls[0]["params"]["start"] == "2017-01-03T00:00:00Z"
    c0, c1, c2 = m["chunks"]
    assert (c0["reused"], c0["malformed_counts_source"]) == (True, "inferred_from_strict_pre_amendment_policy")
    assert (c1["reused"], c1["malformed_counts_source"], c1["n_raw_records_fetched"], c1["n_malformed_excluded"],
            c1["rows"]) == (True, "recorded_in_snapshot_meta", 490, 1, 489)
    assert c1["sha256"] == recorded["sha256"]
    assert (c2["reused"], c2["malformed_counts_source"]) == (False, "measured_this_run")
    assert m["totals"]["n_inferred_chunks"] == 1 and m["totals"]["n_fresh_chunks"] == 2
    assert {p.name: p.read_bytes() for p in saved} == before                        # reused files never rewritten


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
