"""Phase 4C canonical dataset: deterministic canonicalization, dedup rule, hash verification."""

import json
import random
from datetime import datetime, timedelta, timezone

import pytest

from services.historical_news_ingestion import IngestionConfig, run_historical_ingestion
from services.news_alignment import is_eligible
from services.news_dataset import (
    build_canonical_dataset,
    canonicalize,
    dedupe_canonical,
    load_canonical_dataset,
    serialize_dataset,
)
from services.news_schema import NewsArticle, NewsValidationError, historical_information_available_at
from tests.news_fakes import FakeSession, make_provider, page, raw

UTC = timezone.utc
START = datetime(2017, 1, 1, tzinfo=UTC)
END = datetime(2017, 1, 3, tzinfo=UTC)
T = datetime(2017, 1, 2, 15, 0, tzinfo=UTC)


def art(i, created=T, updated=None, headline=None, symbols=("AAPL",), fetched=datetime(2026, 10, 1, tzinfo=UTC)):
    return NewsArticle(provider="alpaca", provider_article_id=str(i), headline=headline or f"H{i}",
                       symbols=symbols, created_at=created, updated_at=updated,
                       information_available_at=historical_information_available_at(created, updated),
                       fetched_at=fetched)


def ingest(tmp_path, *responses, chunk_days=1):
    cfg = IngestionConfig(provider="alpaca", symbols=("AAPL",), start=START, end=END, chunk_days=chunk_days)
    return run_historical_ingestion(make_provider(FakeSession(*responses)), cfg, raw_dir=tmp_path / "raw",
                                    clock=lambda: datetime(2026, 10, 1, 13, tzinfo=UTC))


# ==========================================
# Determinism
# ==========================================


def test_rebuild_from_same_manifest_is_byte_identical(tmp_path):
    manifest = ingest(tmp_path, page([raw(1, created="2017-01-01T10:00:00Z"), raw(2, created="2017-01-01T09:00:00Z")]),
                      page([raw(3), raw(2, created="2017-01-01T09:00:00Z")]))
    a, meta_a = build_canonical_dataset(manifest, tmp_path / "one")
    b, meta_b = build_canonical_dataset(manifest, tmp_path / "two")
    assert a.read_bytes() == b.read_bytes()
    assert meta_a == meta_b
    assert (tmp_path / "one" / (a.stem + ".meta.json")).read_bytes() == \
        (tmp_path / "two" / (b.stem + ".meta.json")).read_bytes()


def test_canonicalization_is_independent_of_input_order():
    records = [art(1), art(2, created=T - timedelta(hours=3)), art(2, created=T - timedelta(hours=3)),
               art(3, updated=T + timedelta(hours=1)), art(3), art(4, symbols=("GOOG",)),
               art(5, created=END), art(6, headline="Same headline"), art(7, headline="Same headline")]
    reference = serialize_dataset(canonicalize(records, symbols=["AAPL"], start=START, end=END)[0])
    rng = random.Random(1)
    for _ in range(20):
        shuffled = records[:]
        rng.shuffle(shuffled)
        out, _ = canonicalize(shuffled, symbols=["AAPL"], start=START, end=END)
        assert serialize_dataset(out) == reference


# ==========================================
# Deduplication rule
# ==========================================


@pytest.mark.parametrize("order", [0, 1])
def test_dedupe_keeps_latest_version_whatever_the_order(order):
    old, new = art(1), art(1, updated=T + timedelta(hours=2), headline="Revised")
    kept, n_dup, n_conflict = dedupe_canonical([old, new] if order == 0 else [new, old])
    assert kept == [new] and (n_dup, n_conflict) == (1, 1)


def test_dedupe_tie_break_is_deterministic():
    a, b = art(1, headline="Alpha"), art(1, headline="Beta")        # identical timestamps
    assert dedupe_canonical([a, b])[0] == dedupe_canonical([b, a])[0] == [a]


def test_identical_duplicates_are_not_conflicts():
    kept, n_dup, n_conflict = dedupe_canonical([art(1), art(1), art(1)])
    assert len(kept) == 1 and (n_dup, n_conflict) == (2, 0)


def test_similar_headlines_are_never_merged():
    out, stats = canonicalize([art(1, headline="Apple beats"), art(2, headline="Apple beats")],
                              symbols=["AAPL"], start=START, end=END)
    assert [a.provider_article_id for a in out] == ["1", "2"] and stats["n_duplicates_removed"] == 0


# ==========================================
# Ingestion interval vs prediction eligibility
# ==========================================


def test_interval_selects_by_publication_but_eligibility_is_unchanged():
    late_revision = art(1, created=T, updated=END + timedelta(days=2))
    out, _ = canonicalize([late_revision], symbols=["AAPL"], start=START, end=END)
    assert out == [late_revision]                                             # in the dataset (published in window)
    assert out[0].information_available_at == END + timedelta(days=2)        # not rewritten
    assert not is_eligible(out[0], END)                                       # still not usable before availability


# ==========================================
# Hash verification
# ==========================================


def test_tampered_snapshot_blocks_build(tmp_path):
    manifest = ingest(tmp_path, page([raw(1, created="2017-01-01T10:00:00Z")]), page([raw(2)]))
    snap = sorted((tmp_path / "raw").glob("*.jsonl"))[0]
    snap.write_bytes(snap.read_bytes().replace(b"Headline", b"Hacked!!"))
    with pytest.raises(NewsValidationError, match="hash mismatch"):
        build_canonical_dataset(manifest, tmp_path / "out")


def test_manifest_hash_mismatch_blocks_build(tmp_path):
    manifest = ingest(tmp_path, page([raw(1, created="2017-01-01T10:00:00Z")]), page([raw(2)]))
    m = json.loads(manifest.read_text(encoding="utf-8"))
    m["chunks"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(NewsValidationError, match="does not match the manifest"):
        build_canonical_dataset(manifest, tmp_path / "out")


def test_manifest_missing_field_rejected(tmp_path):
    bad = tmp_path / "bad.manifest.json"
    bad.write_text(json.dumps({"run_id": "x"}), encoding="utf-8")
    with pytest.raises(NewsValidationError, match="missing"):
        build_canonical_dataset(bad, tmp_path / "out")


def test_canonical_dataset_round_trip_and_tamper_detection(tmp_path):
    manifest = ingest(tmp_path, page([raw(1, created="2017-01-01T10:00:00Z")]), page([raw(2)]))
    path, meta = build_canonical_dataset(manifest, tmp_path / "out")
    articles, loaded_meta = load_canonical_dataset(path)
    assert [a.provider_article_id for a in articles] == ["1", "2"] and loaded_meta == meta

    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(NewsValidationError, match="hash mismatch"):
        load_canonical_dataset(path)
