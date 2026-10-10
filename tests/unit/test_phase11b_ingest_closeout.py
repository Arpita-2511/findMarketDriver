"""
Phase 11B Stage 11B.1 ingestion closeout: synthetic unit tests.

Covers the behavioural contract of training/phase11b_ingest_closeout.py:
  * dev/sealed split at T_dev (<= cutoff is development; > cutoff is sealed)
  * revisions whose updated_at crosses the cutoff land sealed
  * UTX->RTX in-memory mapping with every design-freeze case
  * fail-closed on bad timestamps, conflicting duplicates, invalid keys
  * deterministic, permutation-invariant hashing
  * sealed article content is never read (headline/summary/content untouched)
  * canonical dataset file is never edited
  * no overwrite without explicit authorization
  * reports can be independently verified from the input articles

These tests are Gate A assertions (synthetic only; no real artifacts).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from services.alpaca_news_provider import (
    MALFORMED_ABS_FLOOR,
    MALFORMED_PCT_CEILING,
    MALFORMED_POLICY_AMENDMENT_SHA256,
    MALFORMED_REASONS,
    MALFORMED_RUN_PCT_CAP,
    RUN_CAP_POLICY_AMENDMENT_SHA256,
    malformed_policy,
    run_cap_policy,
)
from services.historical_news_ingestion import malformed_totals
from services.news_schema import NewsArticle, historical_information_available_at
from training.phase11b_ingest_closeout import (
    CloseoutError,
    DEVELOPMENT_CUTOFF_UTC,
    FROZEN_DESIGN_FREEZE_SHA256,
    REPORT_PATH,
    RTX_UTX_BOUNDARY_UTC,
    TICKERS_QUERIED,
    _canonical_json,
    _development_article_sha256,
    _sealed_article_id_sha256,
    map_symbols_for_rtx,
    process,
    run_closeout,
)

UTC = timezone.utc
FETCHED = datetime(2026, 10, 1, tzinfo=UTC)


# ============================================================
# Builders
# ============================================================


def _article(i: int, created_at: datetime, *,
             symbols: tuple = ("AAPL",),
             updated: datetime | None = None,
             headline: str = "Story",
             summary: str | None = None,
             content: str | None = None) -> NewsArticle:
    return NewsArticle(
        provider="alpaca",
        provider_article_id=str(i),
        headline=f"{headline} {i}",
        summary=summary,
        content=content,
        symbols=symbols,
        created_at=created_at,
        updated_at=updated,
        information_available_at=historical_information_available_at(created_at, updated),
        fetched_at=FETCHED,
    )


# ============================================================
# Dev / sealed split (cutoff semantics)
# ============================================================


def test_article_strictly_before_cutoff_is_development():
    a = _article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(seconds=1))
    rpt = process([a])
    assert rpt["development_article_count"] == 1
    assert rpt["sealed_article_count"] == 0


def test_article_exactly_at_cutoff_is_development():
    """<= cutoff is development (frozen boundary rule)."""
    a = _article(1, DEVELOPMENT_CUTOFF_UTC)
    assert a.information_available_at == DEVELOPMENT_CUTOFF_UTC
    rpt = process([a])
    assert rpt["development_article_count"] == 1, "article at exact cutoff must be development"
    assert rpt["sealed_article_count"] == 0


def test_article_one_microsecond_after_cutoff_is_sealed():
    a = _article(1, DEVELOPMENT_CUTOFF_UTC + timedelta(microseconds=1))
    rpt = process([a])
    assert rpt["development_article_count"] == 0
    assert rpt["sealed_article_count"] == 1


def test_revision_pushes_availability_past_cutoff_is_sealed():
    """Created before cutoff; updated after -> information_available_at = updated_at -> sealed."""
    created = DEVELOPMENT_CUTOFF_UTC - timedelta(days=5)
    updated = DEVELOPMENT_CUTOFF_UTC + timedelta(hours=1)
    a = _article(1, created, updated=updated)
    assert a.information_available_at == updated
    rpt = process([a])
    assert rpt["development_article_count"] == 0
    assert rpt["sealed_article_count"] == 1
    assert rpt["created_before_cutoff_available_after_count"] == 1


def test_dev_and_sealed_intersection_empty():
    before = _article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(hours=1))
    at_cutoff = _article(2, DEVELOPMENT_CUTOFF_UTC)
    just_after = _article(3, DEVELOPMENT_CUTOFF_UTC + timedelta(microseconds=1))
    way_after = _article(4, DEVELOPMENT_CUTOFF_UTC + timedelta(days=90))
    rpt = process([before, at_cutoff, just_after, way_after])
    assert rpt["development_article_count"] == 2
    assert rpt["sealed_article_count"] == 2
    assert rpt["development_sealed_intersection_empty"] is True


# ============================================================
# Fail-closed on invalid inputs
# ============================================================


def test_rejects_non_news_article():
    with pytest.raises(CloseoutError, match="not a NewsArticle"):
        process([{"article_key": "x:1"}])  # type: ignore[list-item]


def test_rejects_conflicting_duplicate_ids():
    """Same key, different content -> refuse (don't choose one)."""
    a1 = _article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(days=1), headline="Original")
    a2 = _article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(days=1), headline="Changed")
    assert a1.article_key == a2.article_key
    with pytest.raises(CloseoutError, match="conflicting duplicate"):
        process([a1, a2])


def test_identical_duplicates_are_collapsed():
    a1 = _article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(days=1))
    a2 = _article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(days=1))
    assert a1 == a2
    rpt = process([a1, a2])
    assert rpt["n_articles_total"] == 1
    assert rpt["n_duplicates_collapsed"] == 1


# ============================================================
# UTX -> RTX mapping, every design-freeze case
# ============================================================


def test_map_utx_before_boundary_to_rtx():
    """Case 1: UTX before boundary -> RTX."""
    before = RTX_UTX_BOUNDARY_UTC - timedelta(hours=1)
    assert map_symbols_for_rtx(("UTX",), before) == ("RTX",)


def test_map_rtx_on_or_after_boundary_kept():
    """Case 2: RTX on or after boundary -> RTX."""
    assert map_symbols_for_rtx(("RTX",), RTX_UTX_BOUNDARY_UTC) == ("RTX",)
    assert map_symbols_for_rtx(("RTX",), RTX_UTX_BOUNDARY_UTC + timedelta(hours=1)) == ("RTX",)


def test_map_rtx_before_boundary_excluded():
    """Case 3: RTX before boundary -> excluded (off-period)."""
    before = RTX_UTX_BOUNDARY_UTC - timedelta(hours=1)
    assert map_symbols_for_rtx(("RTX",), before) == ()


def test_map_utx_on_or_after_boundary_excluded():
    """Case 4: UTX on or after boundary -> excluded (off-period)."""
    assert map_symbols_for_rtx(("UTX",), RTX_UTX_BOUNDARY_UTC) == ()
    assert map_symbols_for_rtx(("UTX",), RTX_UTX_BOUNDARY_UTC + timedelta(hours=1)) == ()


def test_map_rtn_only_excluded_from_rtx_but_kept_as_tag():
    """Case 5: RTN alone never substitutes for RTX; RTN stays in the symbol list unchanged."""
    before = RTX_UTX_BOUNDARY_UTC - timedelta(hours=1)
    after = RTX_UTX_BOUNDARY_UTC + timedelta(hours=1)
    assert map_symbols_for_rtx(("RTN",), before) == ("RTN",)
    assert map_symbols_for_rtx(("RTN",), after) == ("RTN",)


def test_map_rtn_plus_utx_before_counted_once_via_utx():
    """Case 6: RTN + UTX before -> RTX added once, UTX removed, RTN kept."""
    before = RTX_UTX_BOUNDARY_UTC - timedelta(hours=1)
    mapped = map_symbols_for_rtx(("RTN", "UTX"), before)
    assert set(mapped) == {"RTN", "RTX"}
    assert mapped.count("RTX") == 1


def test_map_utx_plus_rtx_same_article_counted_once():
    """Case 7: UTX + RTX same article -> RTX appears exactly once."""
    before = RTX_UTX_BOUNDARY_UTC - timedelta(hours=1)
    after = RTX_UTX_BOUNDARY_UTC + timedelta(hours=1)
    for created in (before, after):
        mapped = map_symbols_for_rtx(("UTX", "RTX"), created)
        assert mapped.count("RTX") == 1, f"RTX must appear once; got {mapped}"
        assert "UTX" not in mapped


def test_map_preserves_other_tags():
    """Non-UTX/RTX tags pass through unchanged."""
    before = RTX_UTX_BOUNDARY_UTC - timedelta(hours=1)
    mapped = map_symbols_for_rtx(("AAPL", "UTX", "MSFT"), before)
    assert set(mapped) == {"AAPL", "MSFT", "RTX"}


def test_map_requires_tz_aware_timestamp():
    with pytest.raises(CloseoutError, match="tz-aware"):
        map_symbols_for_rtx(("UTX",), datetime(2020, 1, 1))      # naive


def test_mapping_counts_in_report():
    """Mapping diagnostics bucket every article per the design freeze."""
    before = RTX_UTX_BOUNDARY_UTC - timedelta(hours=1)
    after = RTX_UTX_BOUNDARY_UTC + timedelta(hours=1)
    arts = [
        _article(1, before, symbols=("UTX",)),                 # bucket: utx_before, rtn_only=F
        _article(2, after, symbols=("RTX",)),                  # bucket: rtx_on_or_after
        _article(3, before, symbols=("RTX",)),                 # bucket: rtx_before_excluded
        _article(4, after, symbols=("UTX",)),                  # bucket: utx_on_or_after_excluded
        _article(5, before, symbols=("RTN",)),                 # bucket: rtn_only
        _article(6, before, symbols=("RTN", "UTX")),           # buckets: rtn_plus_utx_before, utx_before
        _article(7, before, symbols=("UTX", "RTX")),           # buckets: utx_and_rtx, utx_before, rtx_before
    ]
    rpt = process(arts)
    c = rpt["rtx_utx_mapping"]["counts"]
    assert c["utx_before_boundary_mapped_to_rtx"] == 3        # arts 1, 6, 7
    assert c["rtx_on_or_after_boundary_kept"] == 1            # art 2
    assert c["rtx_before_boundary_excluded_off_period"] == 2  # arts 3, 7
    assert c["utx_on_or_after_boundary_excluded_off_period"] == 1  # art 4
    assert c["rtn_only_articles"] == 1                        # art 5
    assert c["rtn_plus_utx_before_boundary"] == 1             # art 6
    assert c["utx_and_rtx_same_article"] == 1                 # art 7


# ============================================================
# Hash determinism and independent verification
# ============================================================


def test_development_hash_is_permutation_invariant():
    a1 = _article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(hours=10))
    a2 = _article(2, DEVELOPMENT_CUTOFF_UTC - timedelta(hours=5))
    a3 = _article(3, DEVELOPMENT_CUTOFF_UTC - timedelta(hours=1))
    h_abc = _development_article_sha256([a1, a2, a3])
    h_cba = _development_article_sha256([a3, a2, a1])
    h_bac = _development_article_sha256([a2, a1, a3])
    assert h_abc == h_cba == h_bac


def test_sealed_hash_is_permutation_invariant_and_uses_ids_only():
    a1 = _article(1, DEVELOPMENT_CUTOFF_UTC + timedelta(hours=1))
    a2 = _article(2, DEVELOPMENT_CUTOFF_UTC + timedelta(hours=2))
    keys = [a1.article_key, a2.article_key]
    h1 = _sealed_article_id_sha256(keys)
    h2 = _sealed_article_id_sha256(list(reversed(keys)))
    assert h1 == h2
    expected = hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()
    assert h1 == expected


def test_report_dev_hash_matches_independent_recomputation():
    a1 = _article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(hours=10))
    a2 = _article(2, DEVELOPMENT_CUTOFF_UTC - timedelta(hours=5))
    rpt = process([a2, a1])
    manual = hashlib.sha256(
        "\n".join(sorted(_canonical_json(a.to_dict()) for a in [a1, a2])).encode("utf-8")
    ).hexdigest()
    assert rpt["development_article_sha256"] == manual


def test_report_sealed_hash_matches_independent_recomputation():
    a1 = _article(1, DEVELOPMENT_CUTOFF_UTC + timedelta(hours=1))
    a2 = _article(2, DEVELOPMENT_CUTOFF_UTC + timedelta(hours=2))
    rpt = process([a2, a1])
    manual = hashlib.sha256(
        "\n".join(sorted([a1.article_key, a2.article_key])).encode("utf-8")
    ).hexdigest()
    assert rpt["sealed_article_id_sha256"] == manual


def test_hashes_change_when_a_single_byte_of_content_changes():
    a1 = _article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(hours=1), headline="A")
    rpt1 = process([a1])
    a2 = _article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(hours=1), headline="B")
    rpt2 = process([a2])
    assert rpt1["development_article_sha256"] != rpt2["development_article_sha256"]


def test_empty_input_produces_zero_counts_and_empty_hashes():
    rpt = process([])
    assert rpt["development_article_count"] == 0
    assert rpt["sealed_article_count"] == 0
    expected_empty = hashlib.sha256(b"").hexdigest()
    assert rpt["development_article_sha256"] == expected_empty
    assert rpt["sealed_article_id_sha256"] == expected_empty


# ============================================================
# Report structure (contract fields required by Amendment 03)
# ============================================================


REQUIRED_AMENDMENT_03_KEYS = (
    "split_applied_before_scoring",
    "development_sealed_intersection_empty",
    "development_article_count",
    "sealed_article_count",
    "development_article_sha256",
    "sealed_article_id_sha256",
    "tickers_queried_count",
)


def test_report_contains_every_amendment_03_key():
    a = _article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(days=1))
    rpt = process([a])
    missing = [k for k in REQUIRED_AMENDMENT_03_KEYS if k not in rpt]
    assert not missing, f"Amendment 03 keys missing: {missing}"
    # Static invariants
    assert rpt["split_applied_before_scoring"] is True
    assert rpt["development_sealed_intersection_empty"] is True
    assert rpt["tickers_queried_count"] == 30


def test_report_hashes_are_lowercase_64_hex():
    a = _article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(days=1))
    rpt = process([a])
    for key in ("development_article_sha256", "sealed_article_id_sha256"):
        v = rpt[key]
        assert isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v), (
            f"{key} must be lowercase 64-char hex, got {v!r}"
        )


# ============================================================
# Sealed-content isolation
# ============================================================


def test_sealed_content_never_leaves_closeout():
    """
    The report must never carry sealed headlines, summaries, content, or
    any field that could allow downstream code to read sealed article text.
    """
    sealed = _article(
        99, DEVELOPMENT_CUTOFF_UTC + timedelta(hours=1),
        headline="SEALED_SECRET_HEADLINE",
        summary="SEALED_SECRET_SUMMARY",
    )
    rpt = process([sealed])
    serialized = json.dumps(rpt)
    assert "SEALED_SECRET_HEADLINE" not in serialized, "sealed headline leaked to report"
    assert "SEALED_SECRET_SUMMARY" not in serialized, "sealed summary leaked to report"
    # The sealed article's key is permitted (counting + ID hash semantics)
    assert sealed.article_key == "alpaca:99"


def test_sealed_hash_depends_only_on_id_not_content():
    """Changing sealed article content must not change the sealed ID hash."""
    s1 = _article(99, DEVELOPMENT_CUTOFF_UTC + timedelta(hours=1), headline="X")
    rpt1 = process([s1])
    s2 = _article(99, DEVELOPMENT_CUTOFF_UTC + timedelta(hours=1), headline="Y")
    rpt2 = process([s2])
    assert rpt1["sealed_article_id_sha256"] == rpt2["sealed_article_id_sha256"]


def test_dev_hash_does_not_mix_in_sealed_articles():
    dev = _article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(days=1))
    sealed = _article(2, DEVELOPMENT_CUTOFF_UTC + timedelta(hours=1))
    rpt_dev_only = process([dev])
    rpt_mixed = process([dev, sealed])
    assert rpt_dev_only["development_article_sha256"] == rpt_mixed["development_article_sha256"], (
        "development hash changed when a sealed article was added; sealed must not leak into dev hash"
    )


# ============================================================
# Source immutability (process does not mutate inputs)
# ============================================================


def test_process_does_not_mutate_input_articles():
    arts = [
        _article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(hours=10), symbols=("UTX",),
                 updated=None, headline="A", summary="aa"),
        _article(2, DEVELOPMENT_CUTOFF_UTC + timedelta(hours=1), symbols=("AAPL",)),
    ]
    frozen_before = [(a.article_key, a.symbols, a.created_at, a.information_available_at,
                      a.headline, a.summary, a.content) for a in arts]
    process(arts)
    frozen_after = [(a.article_key, a.symbols, a.created_at, a.information_available_at,
                     a.headline, a.summary, a.content) for a in arts]
    assert frozen_before == frozen_after, "process must not mutate its inputs"


# ============================================================
# run_closeout: file I/O, overwrite behaviour, canonical immutability
# ============================================================


INFERRED = "inferred_from_strict_pre_amendment_policy"


def _chunk(n_raw: int, n_bad: int = 0, source: str = "measured_this_run", reason: str = "missing_or_blank_headline",
           reused: bool = False) -> dict:
    counts = {r: 0 for r in MALFORMED_REASONS}
    counts[reason] += n_bad
    return {"start": "2017-01-01T00:00:00+00:00", "end": "2017-01-31T00:00:00+00:00", "reused": reused,
            "n_raw_records_fetched": n_raw, "n_malformed_excluded": n_bad, "malformed_reason_counts": counts,
            "malformed_fraction": n_bad / max(1, n_raw), "malformed_counts_source": source}


def _write_manifest(tmp_path: Path, chunks: list[dict], *, totals: dict | None = None,
                    policy: dict | None = None, run_cap: dict | None = None, omit_run_cap: bool = False) -> dict:
    """An ingestion manifest in tmp_path/raw shaped like run_historical_ingestion's; returns the meta reference."""
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {"chunks": chunks, "totals": totals if totals is not None else {"chunks": len(chunks), **malformed_totals(chunks)},
                "malformed_policy": policy if policy is not None else malformed_policy()}
    if not omit_run_cap:
        manifest["run_cap_policy"] = run_cap if run_cap is not None else run_cap_policy()
    path = raw / "alpaca_test.manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _closeout(tmp_path: Path, canonical: Path, **kwargs):
    return run_closeout(canonical, raw_dir=tmp_path / "raw", **kwargs)


def _write_canonical_dataset(tmp_path: Path, articles: list[NewsArticle], chunks: list[dict] | None = None,
                             manifest_overrides: dict | None = None) -> Path:
    """Write a minimal canonical dataset + meta, plus the Amendment 04 manifest it references (tmp_path/raw)."""
    from services.news_dataset import serialize_dataset, sort_key, DATASET_VERSION, INTERVAL_RULE, SYMBOL_RULE, DEDUP_RULE
    from services.news_schema import NEWS_SCHEMA_VERSION, HISTORICAL_AVAILABILITY_RULE
    payload = serialize_dataset(sorted(articles, key=sort_key))
    path = tmp_path / "alpaca_test_20170101T000000Z_20260929T000000Z_20261001T000000Z.jsonl"
    path.write_bytes(payload)
    meta = {
        "dataset_version": DATASET_VERSION,
        "schema_version": NEWS_SCHEMA_VERSION,
        "run_id": "alpaca_test_20170101T000000Z_20260929T000000Z_20261001T000000Z",
        "provider": "alpaca",
        "symbols": sorted({s for a in articles for s in a.symbols}),
        "start": "2017-01-01T00:00:00+00:00",
        "end": "2026-09-29T00:00:00+00:00",
        "include_content": False,
        "interval_rule": INTERVAL_RULE,
        "symbol_rule": SYMBOL_RULE,
        "dedup_rule": DEDUP_RULE,
        "availability_rule": HISTORICAL_AVAILABILITY_RULE,
        "manifest": _write_manifest(tmp_path, chunks if chunks is not None else [_chunk(len(articles))],
                                    **(manifest_overrides or {})),
        "source_snapshots": [],
        "n_input_records": len(articles),
        "n_duplicates_removed": 0,
        "n_conflicting_duplicates": 0,
        "n_outside_interval": 0,
        "n_without_requested_symbol": 0,
        "rows": len(articles),
        "first_created_at": sorted(articles, key=sort_key)[0].created_at.isoformat() if articles else None,
        "last_created_at": sorted(articles, key=sort_key)[-1].created_at.isoformat() if articles else None,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_run_closeout_writes_report_and_does_not_touch_canonical(tmp_path):
    arts = [
        _article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(hours=1)),
        _article(2, DEVELOPMENT_CUTOFF_UTC + timedelta(hours=1)),
    ]
    canonical = _write_canonical_dataset(tmp_path, arts)
    canonical_bytes_before = canonical.read_bytes()
    canonical_meta_before = canonical.with_suffix(".meta.json").read_bytes()

    report_path = tmp_path / "news_integrity_report.json"
    out_path, rpt = _closeout(tmp_path, canonical, report_path=report_path)

    assert out_path == report_path and report_path.exists()
    assert canonical.read_bytes() == canonical_bytes_before, "canonical file was modified"
    assert canonical.with_suffix(".meta.json").read_bytes() == canonical_meta_before, "canonical meta was modified"
    assert rpt["development_article_count"] == 1
    assert rpt["sealed_article_count"] == 1
    assert rpt["canonical_input"]["sha256"] == hashlib.sha256(canonical_bytes_before).hexdigest()


def test_run_closeout_refuses_to_overwrite(tmp_path):
    arts = [_article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(hours=1))]
    canonical = _write_canonical_dataset(tmp_path, arts)
    report_path = tmp_path / "report.json"
    _closeout(tmp_path, canonical, report_path=report_path)
    with pytest.raises(CloseoutError, match="Refusing to overwrite"):
        _closeout(tmp_path, canonical, report_path=report_path)


def test_run_closeout_allows_explicit_overwrite(tmp_path):
    arts = [_article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(hours=1))]
    canonical = _write_canonical_dataset(tmp_path, arts)
    report_path = tmp_path / "report.json"
    _closeout(tmp_path, canonical, report_path=report_path)
    # Second call with overwrite=True must succeed and produce identical content
    out1 = report_path.read_bytes()
    _closeout(tmp_path, canonical, report_path=report_path, overwrite=True)
    out2 = report_path.read_bytes()
    # The payload is the same except for `written_at`; parse and compare the body
    j1 = json.loads(out1); j2 = json.loads(out2)
    j1.pop("written_at"); j2.pop("written_at")
    assert j1 == j2


def test_run_closeout_report_passes_test_t5_contract(tmp_path):
    """The written report must satisfy the T5-real assertions."""
    arts = [
        _article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(hours=1)),
        _article(2, DEVELOPMENT_CUTOFF_UTC + timedelta(hours=1)),
    ]
    canonical = _write_canonical_dataset(tmp_path, arts)
    report_path = tmp_path / "report.json"
    _, rpt = _closeout(tmp_path, canonical, report_path=report_path)

    # Mirror the assertions in tests/unit/test_phase11b_leakage.py::_assert_t5_seal_real
    assert rpt["split_applied_before_scoring"] is True
    assert rpt["development_sealed_intersection_empty"] is True
    assert isinstance(rpt["development_article_count"], int) and rpt["development_article_count"] > 0
    assert isinstance(rpt["sealed_article_count"], int) and rpt["sealed_article_count"] >= 0
    for key in ("development_article_sha256", "sealed_article_id_sha256"):
        v = rpt[key]
        assert isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


# ============================================================
# Design freeze pinning
# ============================================================


def test_frozen_design_freeze_sha_constant_matches_the_file():
    """The constant pinned in the module must match the on-disk design freeze."""
    live = hashlib.sha256(
        (Path(__file__).resolve().parents[2] / "data" / "results" / "research" / "phase11" / "11B" / "design_freeze.json").read_bytes()
    ).hexdigest()
    assert live == FROZEN_DESIGN_FREEZE_SHA256


def test_tickers_queried_count_is_30_and_includes_utx_rtx():
    assert len(TICKERS_QUERIED) == 30
    assert "UTX" in TICKERS_QUERIED
    assert "RTX" in TICKERS_QUERIED


# ============================================================
# Amendment 04: malformed-record policy (manifest -> report, Gate B fields)
# ============================================================


AMENDMENT_04 = Path(__file__).resolve().parents[2] / "data" / "results" / "research" / "phase11" / "11B" / \
    "amendment_04_malformed_record_policy.json"


def _two_articles():
    return [_article(1, DEVELOPMENT_CUTOFF_UTC - timedelta(hours=1)),
            _article(2, DEVELOPMENT_CUTOFF_UTC + timedelta(hours=1))]


def test_adopted_amendment_04_matches_code_constants():
    doc = json.loads(AMENDMENT_04.read_text(encoding="utf-8"))
    assert doc["status"] == "ADOPTED" and doc["amendment_id"] == "04"
    assert hashlib.sha256(AMENDMENT_04.read_bytes()).hexdigest() == MALFORMED_POLICY_AMENDMENT_SHA256
    assert set(doc["what_counts_as_malformed"]["reason_categories"]) == set(MALFORMED_REASONS)
    assert tuple(doc["what_counts_as_malformed"]["classification_order"]) == MALFORMED_REASONS
    assert "updated_before_created" not in MALFORMED_REASONS
    assert doc["threshold"]["per_chunk_cap"]["ABS_FLOOR"] == MALFORMED_ABS_FLOOR
    assert doc["threshold"]["per_chunk_cap"]["PCT_CEILING"] == MALFORMED_PCT_CEILING
    assert doc["threshold"]["run_cap"]["RUN_PCT_CAP"] == MALFORMED_RUN_PCT_CAP
    assert doc["applies_to"]["design_freeze_sha256_at_adoption"] == FROZEN_DESIGN_FREEZE_SHA256


def test_report_includes_malformed_summary_fields(tmp_path):
    canonical = _write_canonical_dataset(tmp_path, _two_articles(), chunks=[_chunk(1000, 2)])
    _, rpt = _closeout(tmp_path, canonical, report_path=tmp_path / "r.json")
    assert rpt["n_raw_records_fetched_total"] == 1000
    assert rpt["n_malformed_excluded_total"] == 2
    assert rpt["malformed_fraction_total"] == 0.002                     # exactly at the cap passes (strict >)
    assert rpt["malformed_reason_counts_total"]["missing_or_blank_headline"] == 2
    assert set(rpt["malformed_reason_counts_total"]) == set(MALFORMED_REASONS)
    assert rpt["malformed_policy_amendment_sha256"] == MALFORMED_POLICY_AMENDMENT_SHA256
    assert (rpt["malformed_cap_per_chunk_abs_floor"], rpt["malformed_cap_per_chunk_pct_ceiling"],
            rpt["malformed_cap_run_pct"]) == (10, 0.01, 0.002)
    assert rpt["ingestion_manifest"]["file"] == "alpaca_test.manifest.json"


def test_report_includes_fresh_and_inferred_breakdown(tmp_path):
    chunks = [_chunk(9668, 0, source=INFERRED, reused=True), _chunk(1000, 1)]
    canonical = _write_canonical_dataset(tmp_path, _two_articles(), chunks=chunks)
    _, rpt = _closeout(tmp_path, canonical, report_path=tmp_path / "r.json")
    assert (rpt["n_reused_chunks"], rpt["n_inferred_chunks"], rpt["n_fresh_chunks"]) == (1, 1, 1)
    assert rpt["n_raw_records_fetched_total"] == 10668
    assert rpt["n_raw_records_fetched_among_fresh_chunks"] == 1000
    assert rpt["malformed_fraction_among_fresh_chunks"] == 0.001


def test_closeout_refuses_cumulative_fraction_above_cap(tmp_path):
    canonical = _write_canonical_dataset(tmp_path, _two_articles(), chunks=[_chunk(1000, 3)])   # 0.3% > 0.2%
    with pytest.raises(CloseoutError, match="cumulative malformed fraction"):
        _closeout(tmp_path, canonical, report_path=tmp_path / "r.json")
    assert not (tmp_path / "r.json").exists()


def test_closeout_refuses_fresh_fraction_above_cap_even_when_diluted(tmp_path):
    chunks = [_chunk(100_000, 0, source=INFERRED, reused=True), _chunk(1000, 3)]   # cumulative 0.003%, fresh 0.3%
    canonical = _write_canonical_dataset(tmp_path, _two_articles(), chunks=chunks)
    with pytest.raises(CloseoutError, match="fresh-chunk malformed fraction"):
        _closeout(tmp_path, canonical, report_path=tmp_path / "r.json")
    assert not (tmp_path / "r.json").exists()


def test_closeout_refuses_per_chunk_cap_breach(tmp_path):
    chunks = [_chunk(100_000, 0, source=INFERRED, reused=True), _chunk(500, 11)]   # 11 > max(10, 5)
    canonical = _write_canonical_dataset(tmp_path, _two_articles(), chunks=chunks)
    with pytest.raises(CloseoutError, match="per-chunk malformed cap"):
        _closeout(tmp_path, canonical, report_path=tmp_path / "r.json")


def test_closeout_refuses_unknown_reason_category(tmp_path):
    bad = _chunk(1000, 1)
    bad["malformed_reason_counts"]["updated_before_created"] = 0
    canonical = _write_canonical_dataset(tmp_path, _two_articles(), chunks=[bad])
    with pytest.raises(CloseoutError, match="non-frozen malformed reason"):
        _closeout(tmp_path, canonical, report_path=tmp_path / "r.json")


def test_closeout_refuses_totals_that_disagree_with_chunks(tmp_path):
    chunks = [_chunk(1000, 1)]
    totals = {"chunks": 1, **malformed_totals(chunks), "n_malformed_excluded": 0}
    canonical = _write_canonical_dataset(tmp_path, _two_articles(), chunks=chunks,
                                         manifest_overrides={"totals": totals})
    with pytest.raises(CloseoutError, match="totals disagree"):
        _closeout(tmp_path, canonical, report_path=tmp_path / "r.json")


def test_closeout_refuses_policy_mismatch(tmp_path):
    policy = {**malformed_policy(), "amendment_sha256": "0" * 64}
    canonical = _write_canonical_dataset(tmp_path, _two_articles(), manifest_overrides={"policy": policy})
    with pytest.raises(CloseoutError, match="malformed_policy"):
        _closeout(tmp_path, canonical, report_path=tmp_path / "r.json")


def test_closeout_refuses_missing_or_tampered_manifest(tmp_path):
    canonical = _write_canonical_dataset(tmp_path, _two_articles())
    manifest = tmp_path / "raw" / "alpaca_test.manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(CloseoutError, match="manifest hash mismatch"):
        _closeout(tmp_path, canonical, report_path=tmp_path / "r.json")
    manifest.unlink()
    with pytest.raises(CloseoutError, match="manifest referenced by the canonical dataset not found"):
        _closeout(tmp_path, canonical, report_path=tmp_path / "r.json")


def test_closeout_refuses_raw_total_below_article_count(tmp_path):
    canonical = _write_canonical_dataset(tmp_path, _two_articles(), chunks=[_chunk(1)])
    with pytest.raises(CloseoutError, match="raw record total is smaller"):
        _closeout(tmp_path, canonical, report_path=tmp_path / "r.json")


def test_closeout_report_satisfies_gate_b_amendment_04_checks(tmp_path):
    """Mirror of the Amendment 04 assertions in _assert_t5_seal_real, on a passing synthetic report."""
    doc = json.loads(AMENDMENT_04.read_text(encoding="utf-8"))
    canonical = _write_canonical_dataset(tmp_path, _two_articles(),
                                         chunks=[_chunk(9668, 0, source=INFERRED, reused=True), _chunk(2000, 4)])
    _, rpt = _closeout(tmp_path, canonical, report_path=tmp_path / "r.json")
    cap = doc["threshold"]["run_cap"]["RUN_PCT_CAP"]
    assert rpt["malformed_policy_amendment_sha256"] == hashlib.sha256(AMENDMENT_04.read_bytes()).hexdigest()
    assert rpt["n_raw_records_fetched_total"] >= rpt["development_article_count"] + rpt["sealed_article_count"]
    assert rpt["malformed_fraction_total"] <= cap and rpt["malformed_fraction_among_fresh_chunks"] <= cap
    assert set(rpt["malformed_reason_counts_total"]) <= set(doc["what_counts_as_malformed"]["reason_categories"])


# ============================================================
# Amendment 05: run caps judged on final totals (run-level policy identity)
# ============================================================


AMENDMENT_05 = AMENDMENT_04.parent / "amendment_05_final_totals_run_cap.json"


def test_adopted_amendment_05_matches_code_constants():
    doc = json.loads(AMENDMENT_05.read_text(encoding="utf-8"))
    a04 = json.loads(AMENDMENT_04.read_text(encoding="utf-8"))
    assert doc["status"] == "ADOPTED" and doc["amendment_id"] == "05"
    assert hashlib.sha256(AMENDMENT_05.read_bytes()).hexdigest() == RUN_CAP_POLICY_AMENDMENT_SHA256
    assert doc["applies_to"]["amendment_04_sha256"] == MALFORMED_POLICY_AMENDMENT_SHA256 \
        == hashlib.sha256(AMENDMENT_04.read_bytes()).hexdigest()
    assert doc["applies_to"]["design_freeze_sha256"] == FROZEN_DESIGN_FREEZE_SHA256
    unchanged = doc["policy_change"]["unchanged"]
    assert tuple(unchanged["classification_order"]) == MALFORMED_REASONS \
        == tuple(a04["what_counts_as_malformed"]["classification_order"])
    assert (unchanged["ABS_FLOOR"], unchanged["PCT_CEILING"], unchanged["RUN_PCT_CAP"]) == \
        (MALFORMED_ABS_FLOOR, MALFORMED_PCT_CEILING, MALFORMED_RUN_PCT_CAP) == \
        (a04["threshold"]["per_chunk_cap"]["ABS_FLOOR"], a04["threshold"]["per_chunk_cap"]["PCT_CEILING"],
         a04["threshold"]["run_cap"]["RUN_PCT_CAP"])
    assert run_cap_policy() == {"amendment_sha256": RUN_CAP_POLICY_AMENDMENT_SHA256, "evaluation": "final_totals",
                                "run_pct_cap": 0.002}


def test_report_records_run_cap_policy(tmp_path):
    canonical = _write_canonical_dataset(tmp_path, _two_articles(), chunks=[_chunk(1000, 1)])
    _, rpt = _closeout(tmp_path, canonical, report_path=tmp_path / "r.json")
    assert rpt["run_cap_policy_amendment_sha256"] == RUN_CAP_POLICY_AMENDMENT_SHA256
    assert rpt["run_cap_evaluation"] == "final_totals"
    assert rpt["malformed_policy_amendment_sha256"] == MALFORMED_POLICY_AMENDMENT_SHA256      # unchanged


def test_closeout_refuses_manifest_without_run_cap_policy(tmp_path):
    canonical = _write_canonical_dataset(tmp_path, _two_articles(), manifest_overrides={"omit_run_cap": True})
    with pytest.raises(CloseoutError, match="run_cap_policy is missing or differs"):
        _closeout(tmp_path, canonical, report_path=tmp_path / "r.json")
    assert not (tmp_path / "r.json").exists()


def test_closeout_refuses_wrong_run_cap_hash(tmp_path):
    canonical = _write_canonical_dataset(tmp_path, _two_articles(),
                                         manifest_overrides={"run_cap": {**run_cap_policy(), "amendment_sha256": "0" * 64}})
    with pytest.raises(CloseoutError, match="run_cap_policy is missing or differs"):
        _closeout(tmp_path, canonical, report_path=tmp_path / "r.json")
    assert not (tmp_path / "r.json").exists()


def test_closeout_refuses_evaluation_other_than_final_totals(tmp_path):
    canonical = _write_canonical_dataset(tmp_path, _two_articles(),
                                         manifest_overrides={"run_cap": {**run_cap_policy(), "evaluation": "per_chunk"}})
    with pytest.raises(CloseoutError, match="run_cap_policy is missing or differs"):
        _closeout(tmp_path, canonical, report_path=tmp_path / "r.json")
    assert not (tmp_path / "r.json").exists()


def test_closeout_refuses_missing_or_modified_amendment_05_file(tmp_path):
    import training.phase11b_ingest_closeout as closeout_module
    canonical = _write_canonical_dataset(tmp_path, _two_articles())
    modified = tmp_path / "amendment_05_modified.json"
    modified.write_bytes(AMENDMENT_05.read_bytes() + b" ")
    original = closeout_module.AMENDMENT_05_PATH
    try:
        for stand_in in (modified, tmp_path / "does_not_exist.json"):
            closeout_module.AMENDMENT_05_PATH = stand_in
            with pytest.raises(CloseoutError, match="amendment_05_final_totals_run_cap.json is missing or differs"):
                _closeout(tmp_path, canonical, report_path=tmp_path / "r.json")
            assert not (tmp_path / "r.json").exists()
    finally:
        closeout_module.AMENDMENT_05_PATH = original


def test_closeout_judges_final_totals_not_the_first_fresh_chunk(tmp_path):
    """The incident pattern: 1/490 in the first fresh chunk, then clean fresh chunks -> final 1/550 passes."""
    chunks = [_chunk(9668, 0, source=INFERRED, reused=True), _chunk(490, 1, source="recorded_in_snapshot_meta",
                                                                    reused=True), _chunk(60)]
    canonical = _write_canonical_dataset(tmp_path, _two_articles(), chunks=chunks)
    _, rpt = _closeout(tmp_path, canonical, report_path=tmp_path / "r.json")
    assert rpt["n_raw_records_fetched_among_fresh_chunks"] == 550 and rpt["n_inferred_chunks"] == 1
    assert rpt["malformed_fraction_among_fresh_chunks"] == 1 / 550
    assert rpt["malformed_fraction_total"] == 1 / 10218
