"""
Phase 8 backfill mechanics (offline, mocked provider, fake FinBERT):
resumable chunked ingestion, resumable sentiment scoring, audit, and an
end-to-end multi-chunk run into the unchanged Phase 5C / 6 / 7 pipelines.
"""

import json
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from services.finbert_sentiment import SentimentModelError
from services.historical_events import run_historical_events
from services.historical_news_ingestion import IngestionConfig, find_reusable_snapshot, run_historical_ingestion
from services.historical_sentiment import (
    PINNED_FINBERT_REVISION,
    HistoricalSentimentError,
    backend_provenance,
    load_checkpoint,
    run_historical_sentiment,
    score_canonical_articles,
    serialize_records,
)
from services.news_audit import audit_canonical_dataset
from services.news_dataset import build_canonical_dataset
from services.news_provider import NewsProviderError, NewsQuery
from services.news_schema import NewsValidationError
from tests.conftest import calendar_2024
from tests.news_fakes import FakeResponse, FakeSession, make_provider, page, raw
from tests.sentiment_fakes import bars_for, fake_service, news, write_canonical
from training.build_dataset import save_raw_snapshot
from training.feature_store import build_feature_store, load_feature_store

UTC = timezone.utc
NY = ZoneInfo("America/New_York")
START, END = datetime(2017, 1, 1, tzinfo=UTC), datetime(2017, 1, 4, tzinfo=UTC)
CLOCK = datetime(2026, 10, 1, 13, 0, tzinfo=UTC)


def cfg(**overrides):
    base = dict(provider="alpaca", symbols=("AAPL",), start=START, end=END, chunk_days=1)
    base.update(overrides)
    return IngestionConfig(**base)


def chunk_pages():
    """One page per daily chunk (2017-01-01, 01-02, 01-03)."""
    return [page([raw(1, created="2017-01-01T15:00:00Z")]), page([raw(2, created="2017-01-02T15:00:00Z")]),
            page([raw(3, created="2017-01-03T15:00:00Z")])]


CLOCK2 = datetime(2026, 10, 2, 9, 0, tzinfo=UTC)         # a later run (different run_id / manifest)


def run(raw_dir, session, clock=CLOCK, **kw):
    return run_historical_ingestion(make_provider(session), cfg(**kw.pop("config", {})), raw_dir=raw_dir,
                                    clock=lambda: clock, **kw)


# ==========================================
# Resumable ingestion
# ==========================================


def test_resume_reuses_completed_chunks_after_a_failure(tmp_path):
    raw_dir = tmp_path / "raw"
    pages = chunk_pages()
    with pytest.raises(NewsProviderError):                                     # chunk 2 fails
        run(raw_dir, FakeSession(pages[0], *[FakeResponse(500)] * 4))
    assert not list(raw_dir.glob("*.manifest.json"))

    second = FakeSession(pages[1], pages[2])                                  # serves only chunks 2 and 3
    manifest = run(raw_dir, second, resume=True)
    m = json.loads(manifest.read_text(encoding="utf-8"))
    assert [c["reused"] for c in m["chunks"]] == [True, False, False]
    assert m["totals"]["n_reused_chunks"] == 1 and len(second.calls) == 2
    assert second.calls[0]["params"]["start"] == "2017-01-02T00:00:00Z"

    # identical canonical dataset to an uninterrupted run
    clean = run(tmp_path / "clean", FakeSession(*chunk_pages()))
    a, _ = build_canonical_dataset(manifest, tmp_path / "out_a")
    b, _ = build_canonical_dataset(clean, tmp_path / "out_b")
    assert a.read_bytes() == b.read_bytes()


def test_without_resume_nothing_is_reused(tmp_path):
    raw_dir = tmp_path / "raw"
    run(raw_dir, FakeSession(*chunk_pages()))
    session = FakeSession(*chunk_pages())
    with pytest.raises(FileExistsError, match="snapshot already exists"):     # refetched, then refused to overwrite
        run(raw_dir, session, clock=CLOCK2)
    assert len(session.calls) == 1                                             # chunk 1 was fetched again


def test_snapshots_with_different_request_are_not_reused(tmp_path):
    raw_dir = tmp_path / "raw"
    run(raw_dir, FakeSession(*chunk_pages()), config={"page_size": 10})
    query = NewsQuery(symbols=("AAPL",), start=START, end=datetime(2017, 1, 2, tzinfo=UTC))   # default page size
    assert find_reusable_snapshot(raw_dir, "alpaca", query) is None
    query10 = NewsQuery(symbols=("AAPL",), start=START, end=datetime(2017, 1, 2, tzinfo=UTC), page_size=10)
    assert find_reusable_snapshot(raw_dir, "alpaca", query10) is not None
    no_content = NewsQuery(symbols=("AAPL",), start=START, end=datetime(2017, 1, 2, tzinfo=UTC),
                           page_size=10, include_content=False)
    assert find_reusable_snapshot(raw_dir, "alpaca", no_content) is None


def test_tampered_snapshot_is_not_silently_refetched(tmp_path):
    raw_dir = tmp_path / "raw"
    run(raw_dir, FakeSession(*chunk_pages()))
    first = sorted(p for p in raw_dir.glob("*.jsonl"))[0]
    first.write_bytes(first.read_bytes().replace(b"Headline", b"Hacked!!"))
    with pytest.raises(NewsValidationError, match="hash mismatch"):
        run(raw_dir, FakeSession(), clock=CLOCK2, resume=True)


def test_snapshot_without_metadata_is_ignored(tmp_path):
    raw_dir = tmp_path / "raw"
    run(raw_dir, FakeSession(*chunk_pages()))
    first = sorted(raw_dir.glob("*.jsonl"))[0]
    first.with_suffix(".meta.json").unlink()                                  # interrupted write
    query = NewsQuery(symbols=("AAPL",), start=START, end=datetime(2017, 1, 2, tzinfo=UTC))
    assert find_reusable_snapshot(raw_dir, "alpaca", query) is None


# ==========================================
# Resumable sentiment scoring
# ==========================================

ARTS = [news(i, datetime(2024, 7, 8 + i % 4, 10, tzinfo=NY), ["Apple beats", "Apple plunge", "Apple meeting"][i % 3])
        for i in range(10)]


class Interrupt(Exception):
    pass


def interrupting_service(after_calls):
    svc = fake_service(batch_size=2)
    original = svc.backend.predict_logits

    def predict(texts):
        if len(svc.backend.calls) >= after_calls:
            raise Interrupt("simulated crash")
        return original(texts)

    svc.backend.predict_logits = predict
    return svc


def test_checkpoint_resume_scores_only_the_rest(tmp_path, monkeypatch):
    import services.historical_sentiment as hs
    monkeypatch.setattr(hs, "CHECKPOINT_BLOCK", 4)
    ckpt = tmp_path / "ckpt.jsonl"

    with pytest.raises(SentimentModelError, match="inference failed"):        # crash in block 2
        score_canonical_articles(ARTS, interrupting_service(after_calls=2), checkpoint=ckpt)
    assert len(load_checkpoint(ckpt, hs.backend_provenance(fake_service()))) == 4

    resumed = fake_service(batch_size=2)
    scored, stats = score_canonical_articles(ARTS, resumed, checkpoint=ckpt)
    assert stats["n_from_checkpoint"] == 4 and sum(len(c) for c in resumed.backend.calls) == 6
    reference, _ = score_canonical_articles(ARTS, fake_service())
    assert serialize_records(scored) == serialize_records(reference)


def test_checkpoint_torn_last_line_is_repaired(tmp_path):
    ckpt = tmp_path / "ckpt.jsonl"
    score_canonical_articles(ARTS[:3], fake_service(), checkpoint=ckpt)
    ckpt.write_text(ckpt.read_text(encoding="utf-8") + '{"record_version": "sentiment_rec', encoding="utf-8")
    assert len(load_checkpoint(ckpt, backend_provenance(fake_service()))) == 3
    assert ckpt.read_text(encoding="utf-8").endswith("\n")


def test_checkpoint_from_another_revision_is_rejected(tmp_path):
    ckpt = tmp_path / "ckpt.jsonl"
    score_canonical_articles(ARTS[:3], fake_service(revision="revA"), checkpoint=ckpt)
    with pytest.raises(HistoricalSentimentError, match="mixed sentiment provenance"):
        score_canonical_articles(ARTS, fake_service(revision="revB"), checkpoint=ckpt)


def test_checkpoint_entry_for_changed_text_is_not_reused(tmp_path):
    ckpt = tmp_path / "ckpt.jsonl"
    score_canonical_articles([ARTS[0]], fake_service(), checkpoint=ckpt)
    edited = news(0, ARTS[0].created_at, "Apple shares plunge")               # same id, different text
    svc = fake_service()
    scored, stats = score_canonical_articles([edited], svc, checkpoint=ckpt)
    assert stats["n_from_checkpoint"] == 0 and len(svc.backend.calls) == 1
    assert scored[0].sentiment.label == "negative"


def test_cli_defaults_to_the_pinned_revision():
    assert PINNED_FINBERT_REVISION == "4556d13015211d73dccd3fdd39d39232506f3e43"


# ==========================================
# Audit
# ==========================================


def test_audit_reports_without_deleting(tmp_path):
    a1 = news(1, datetime(2017, 1, 3, 10, tzinfo=NY), "Same story")
    a2 = news(2, datetime(2017, 1, 3, 10, tzinfo=NY), "Same story")            # same headline+time, other id
    a3 = news(3, datetime(2017, 1, 6, 17, 30, tzinfo=NY), "After close")        # after 16:00
    a4 = news(4, datetime(2017, 1, 7, 11, tzinfo=NY), "Saturday story")
    path = write_canonical(tmp_path, [a1, a2, a3, a4], start=datetime(2017, 1, 1, tzinfo=UTC),
                           end=datetime(2017, 3, 1, tzinfo=UTC))
    r = audit_canonical_dataset(path)
    assert r["articles"] == 4 and r["duplicate_article_ids"] == 0
    assert r["same_headline_and_created_at_different_ids"] == 1
    assert r["available_after_close_or_weekend"] == 2
    assert r["months_in_interval"] == 2 and r["months_without_articles"] == ["2017-02"]
    assert r["articles_per_year"] == {2017: 4} and r["with_content"] == 0


# ==========================================
# End to end: multi-chunk backfill -> 5C -> 6 -> 7 (unchanged pipelines)
# ==========================================


def test_multi_chunk_backfill_flows_into_existing_pipelines(tmp_path):
    sessions = [d for d in pd.bdate_range("2024-06-24", "2024-07-31").date if calendar_2024().is_session(d)]
    bars = bars_for(sessions)
    start, end = datetime(2024, 7, 1, tzinfo=UTC), datetime(2024, 7, 13, tzinfo=UTC)
    pages = [page([raw(10 + i, created=f"2024-07-{1 + 3 * i:02d}T15:00:00Z", headline=h)])
             for i, h in enumerate(["Apple Reports Q1 EPS beat", "Apple Says It Is Suing Qualcomm",
                                    "Wells Fargo Cuts Apple Estimates", "Inside The Hidden Economy Of Pawn Shops"])]
    manifest = run_historical_ingestion(make_provider(FakeSession(*pages)),
                                        cfg(start=start, end=end, chunk_days=3), raw_dir=tmp_path / "raw",
                                        clock=lambda: CLOCK)
    canonical, cmeta = build_canonical_dataset(manifest, tmp_path / "news")
    assert cmeta["rows"] == 4

    source = {"file": "fixture", "sha256": "0" * 64}
    s = run_historical_sentiment(canonical, bars, fake_service(), calendar_source=source,
                                 output_dir=tmp_path / "sentiment", checkpoint=tmp_path / "ckpt.jsonl")
    e = run_historical_events(canonical, s["sentiment_path"], bars, calendar_source=source,
                              output_dir=tmp_path / "events")
    assert s["daily_meta"]["rows"] == e["daily_meta"]["rows"] == 9            # sessions 07-01 .. 07-12

    from tests.feature_store_fakes import write_technical
    technical = write_technical(tmp_path / "tech", dates=sessions[:-1])
    snapshot = save_raw_snapshot(bars, "AAPL", "1y", CLOCK, raw_dir=tmp_path / "stocks")
    path, meta = build_feature_store(technical, s["daily_path"], e["daily_path"], snapshot, tmp_path / "features")
    store, _ = load_feature_store(path)
    q = meta["data_quality"]
    assert q["coverage"]["all_three"] == 9 and q["duplicate_keys"] == 0 and q["infinite_values"] == 0
    row = store[store["trading_date"] == date(2024, 7, 4 + 1)].iloc[0]        # 07-05: articles of 07-01 and 07-04
    assert row["sentiment__news_count"] == 2
