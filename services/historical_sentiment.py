"""
Historical FinBERT scoring and daily sentiment dataset (Phase 5C).

    canonical news dataset (Phase 4C, hash-verified)
        -> unique_articles           identical duplicates collapse, conflicts raise
        -> score_canonical_articles  Phase 5A FinBertSentimentService (batched)
                                     -> ScoredArticle (Phase 5B validation)
        -> save_sentiment_dataset    <source>.finbert-<rev>.sentiment.jsonl + .meta.json
        -> load_sentiment_dataset    re-verified, re-joined to the canonical articles
        -> build_daily_rows          Phase 5B generate_sentiment_features (unchanged)
        -> save_daily_dataset        <source>.finbert-<rev>.daily.jsonl + .meta.json

This module orchestrates only: text policy, inference, validation and
aggregation all come from services/finbert_sentiment.py and
services/sentiment_features.py.

Article-level records hold identity, symbols, information_available_at and
the sentiment fields - never article text. Reloading re-joins them to the
canonical dataset, whose sha256 must match the one recorded at scoring time.

One provenance per dataset: every result must share model_name,
model_revision, inference_version and text_policy with the scoring backend;
mixing fails. The model revision is part of the output file name.

Content identity vs run metadata: the output FILE (and its sha256) contains
only deterministic content; generated_at lives in the .meta.json only.
Same canonical input + same model revision + same settings -> same logical
records (float values may differ in the last bits across hardware / torch
builds; the file hash then differs and an existing file is NOT overwritten).

Leakage: scoring decides nothing about eligibility. Daily rows use only the
Phase 5B rule information_available_at <= prediction_timestamp (D 16:30 New York).

FinBERT sentiment is an NLP signal about article text; it does not
establish that the news caused any price movement.

Usage (from the project root; no API keys needed - news is already local):

    python -m services.historical_sentiment ^
        --news data/processed/news/<run_id>.jsonl ^
        --market-snapshot data/raw/stocks/<snapshot>.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.finbert_sentiment import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    INFERENCE_VERSION,
    TEXT_POLICY,
    FinBertSentimentService,
    SentimentModelError,
    SentimentResult,
    build_sentiment_text,
    text_hash,
)
from services.market_calendar_service import TradingCalendar  # noqa: E402
from services.news_alignment import check_not_future  # noqa: E402
from services.news_dataset import load_canonical_dataset  # noqa: E402
from services.news_schema import NewsArticle, parse_timestamp  # noqa: E402
from services.sentiment_features import (  # noqa: E402
    SENTIMENT_FEATURE_VERSION,
    ScoredArticle,
    SentimentFeatureError,
    SentimentFeatureRow,
    generate_sentiment_features,
    prepare_scored_articles,
)
from training.build_dataset import file_sha256, load_raw_snapshot, meta_path_for  # noqa: E402

SENTIMENT_DIR = PROJECT_ROOT / "data" / "processed" / "sentiment"
SENTIMENT_RECORD_VERSION = "sentiment_records_v1"
DAILY_DATASET_VERSION = "daily_sentiment_v1"
RECORD_ORDER = "information_available_at, provider, provider_article_id"
DAILY_ORDER = "symbol, trading_date"

# Phase 8: the FinBERT commit this project's datasets were produced with (CLI default).
PINNED_FINBERT_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
CHECKPOINT_BLOCK = 512           # articles scored between checkpoint flushes

PROVENANCE_FIELDS = ("model_name", "model_revision", "inference_version", "text_policy")


class HistoricalSentimentError(ValueError):
    """Invalid input, provenance or stored sentiment data."""


# ==========================================
# Provenance
# ==========================================


def backend_provenance(service: FinBertSentimentService) -> dict[str, str]:
    """The single provenance every result of a run must carry."""
    return _require_provenance({
        "model_name": service.backend.model_name,
        "model_revision": service.backend.model_revision,
        "inference_version": INFERENCE_VERSION,
        "text_policy": TEXT_POLICY,
    })


def _require_provenance(prov: dict) -> dict[str, str]:
    missing = [k for k in PROVENANCE_FIELDS if not isinstance(prov.get(k), str) or not prov[k].strip()]
    if missing:
        raise HistoricalSentimentError(f"missing sentiment provenance: {missing}")
    return {k: prov[k] for k in PROVENANCE_FIELDS}


def check_consistent_provenance(results: Iterable[SentimentResult], expected: dict[str, str]) -> None:
    """Every result must match `expected` exactly - no mixing of models/revisions/policies."""
    expected = _require_provenance(expected)
    for r in results:
        actual = {k: getattr(r, k) for k in PROVENANCE_FIELDS}
        _require_provenance(actual)
        if actual != expected:
            diff = {k: (actual[k], expected[k]) for k in PROVENANCE_FIELDS if actual[k] != expected[k]}
            raise HistoricalSentimentError(f"mixed sentiment provenance in one dataset: {diff}")


# ==========================================
# Scoring
# ==========================================


def _article_order(a: NewsArticle):
    return (a.information_available_at, a.provider, a.provider_article_id)


def unique_articles(articles: Iterable[NewsArticle]) -> tuple[list[NewsArticle], int]:
    """
    Collapse identical duplicates of (provider, provider_article_id); raise on
    conflicting ones (no headline-based merging). Sorted by RECORD_ORDER.
    Returns (articles, number of duplicates collapsed).
    """
    seen: dict[tuple[str, str], NewsArticle] = {}
    collapsed = 0
    for a in articles:
        if not isinstance(a, NewsArticle):
            raise HistoricalSentimentError("canonical input must contain NewsArticle records")
        existing = seen.get(a.key)
        if existing is None:
            seen[a.key] = a
        elif existing == a:
            collapsed += 1
        else:
            raise HistoricalSentimentError(f"conflicting canonical records for {a.article_key}")
    return sorted(seen.values(), key=_article_order), collapsed


def load_checkpoint(path: Path, provenance: dict) -> dict:
    """
    Phase 8 resumable scoring: read a checkpoint (sentiment_records_v1 lines)
    into {((provider, id), input_text_hash): SentimentResult}. Every record is
    re-validated (Phase 5A contract) and must carry exactly `provenance`.
    A partially written LAST line (interrupted append) is removed; any other
    malformed line raises.
    """
    path = Path(path)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if text and not text.endswith("\n"):
        lines = lines[:-1]                                         # drop the torn tail ...
        path.write_text("".join(line + "\n" for line in lines if line), encoding="utf-8", newline="\n")
    cached = {}
    for line in lines:
        if not line.strip():
            continue
        record = SentimentRecord.from_dict(json.loads(line))
        check_consistent_provenance([record.sentiment], provenance)
        cached[(record.key, record.sentiment.input_text_hash)] = record.sentiment
    return cached


def _append_checkpoint(path: Path, scored: list[ScoredArticle]) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for s in scored:
            f.write(_dumps(SentimentRecord.from_scored(s).to_dict()) + "\n")
        f.flush()
        os.fsync(f.fileno())


def score_canonical_articles(articles: Iterable[NewsArticle], service: FinBertSentimentService, *,
                             checkpoint: Path | None = None) -> tuple[list[ScoredArticle], dict]:
    """
    Score each unique article once with the Phase 5A service. Impossible-future
    articles are rejected BEFORE inference. Returns (scored articles in
    RECORD_ORDER, stats).

    checkpoint (Phase 8): results are appended to this file every
    CHECKPOINT_BLOCK articles; a rerun reuses every checkpointed result whose
    (article key, text_v1 hash) and provenance match, and scores only the rest.
    The final dataset is identical either way.
    """
    articles = list(articles)
    unique, collapsed = unique_articles(articles)
    for a in unique:
        check_not_future(a)                          # FutureInformationError, before any model work

    provenance = backend_provenance(service)
    cached = load_checkpoint(checkpoint, provenance) if checkpoint is not None else {}
    results: dict[tuple[str, str], SentimentResult] = {}
    for a in unique:
        hit = cached.get((a.key, text_hash(build_sentiment_text(a))))
        if hit is not None:
            results[a.key] = hit
    todo = [a for a in unique if a.key not in results]

    block = CHECKPOINT_BLOCK if checkpoint is not None else max(len(todo), 1)
    for start in range(0, len(todo), block):
        part = todo[start:start + block]
        fresh = service.score_articles(part)          # batched; order == input order
        if len(fresh) != len(part):
            raise HistoricalSentimentError("sentiment service returned a different number of results")
        check_consistent_provenance(fresh, provenance)
        try:
            scored_part = [ScoredArticle(a, r) for a, r in zip(part, fresh)]
        except SentimentFeatureError as e:
            raise HistoricalSentimentError(str(e)) from None
        if checkpoint is not None:
            _append_checkpoint(checkpoint, scored_part)
        results.update({s.article.key: s.sentiment for s in scored_part})

    try:
        scored = [ScoredArticle(a, results[a.key]) for a in unique]
    except SentimentFeatureError as e:
        raise HistoricalSentimentError(str(e)) from None
    stats = {"n_input_articles": len(articles), "n_duplicates_collapsed": collapsed,
             "n_scored_articles": len(scored), "n_from_checkpoint": len(unique) - len(todo)}
    return prepare_scored_articles(scored), stats


# ==========================================
# Article-level records
# ==========================================


@dataclass(frozen=True)
class SentimentRecord:
    """Persisted form of one ScoredArticle (no article text)."""

    provider: str
    provider_article_id: str
    symbols: tuple[str, ...]
    information_available_at: datetime
    sentiment: SentimentResult

    @classmethod
    def from_scored(cls, item: ScoredArticle) -> "SentimentRecord":
        a = item.article
        return cls(a.provider, a.provider_article_id, a.symbols, a.information_available_at, item.sentiment)

    @property
    def key(self) -> tuple[str, str]:
        return (self.provider, self.provider_article_id)

    def to_dict(self) -> dict:
        return {
            "record_version": SENTIMENT_RECORD_VERSION,
            "provider": self.provider,
            "provider_article_id": self.provider_article_id,
            "symbols": list(self.symbols),
            "information_available_at": self.information_available_at.isoformat(),
            **self.sentiment.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SentimentRecord":
        required = ("record_version", "provider", "provider_article_id", "symbols", "information_available_at",
                    "label", "positive_probability", "negative_probability", "neutral_probability",
                    "sentiment_score", "input_text_hash", *PROVENANCE_FIELDS)
        missing = [k for k in required if d.get(k) in (None, "")]
        if missing:
            raise HistoricalSentimentError(f"sentiment record is missing {missing}")
        if d["record_version"] != SENTIMENT_RECORD_VERSION:
            raise HistoricalSentimentError(f"unsupported record_version {d['record_version']!r}")
        try:
            sentiment = SentimentResult(**{k: d[k] for k in (
                "inference_version", "model_name", "model_revision", "label", "positive_probability",
                "negative_probability", "neutral_probability", "sentiment_score", "input_text_hash",
                "text_policy")})
        except (SentimentModelError, TypeError) as e:            # Phase 5A validation
            raise HistoricalSentimentError(
                f"invalid sentiment record {d['provider']}:{d['provider_article_id']}: {e}") from None
        return cls(d["provider"], str(d["provider_article_id"]), tuple(d["symbols"]),
                   parse_timestamp(d["information_available_at"], "information_available_at"), sentiment)


def _dumps(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def serialize_records(scored: Iterable[ScoredArticle]) -> bytes:
    return "".join(_dumps(SentimentRecord.from_scored(s).to_dict()) + "\n" for s in scored).encode("utf-8")


def _write_once(path: Path, payload: bytes) -> None:
    """Write, or accept an identical existing file; never overwrite different content."""
    if path.exists() and path.read_bytes() != payload:
        raise FileExistsError(f"{path.name} already exists with different content; refusing to overwrite")
    path.write_bytes(payload)


def _write_meta(path: Path, meta: dict) -> None:
    with open(meta_path_for(path), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")


def _revision_tag(revision: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", revision)[:12] or "unknown"


def sentiment_output_path(canonical_path: Path, provenance: dict, output_dir: Path = SENTIMENT_DIR) -> Path:
    stem = Path(canonical_path).name.removesuffix(".jsonl")
    return Path(output_dir) / f"{stem}.finbert-{_revision_tag(provenance['model_revision'])}.sentiment.jsonl"


def save_sentiment_dataset(scored: list[ScoredArticle], stats: dict, provenance: dict, canonical_path: Path,
                           canonical_meta: dict, output_dir: Path = SENTIMENT_DIR,
                           generated_at: datetime | None = None) -> tuple[Path, dict]:
    provenance = _require_provenance(provenance)
    check_consistent_provenance((s.sentiment for s in scored), provenance)
    ordered = prepare_scored_articles(scored)                  # deterministic order, duplicate safety
    payload = serialize_records(ordered)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = sentiment_output_path(canonical_path, provenance, output_dir)
    _write_once(path, payload)

    meta = {
        "record_version": SENTIMENT_RECORD_VERSION,
        "source_dataset": {"file": Path(canonical_path).name, "sha256": canonical_meta.get("sha256"),
                           "run_id": canonical_meta.get("run_id")},
        "provenance": provenance,
        **stats,
        "rows": len(ordered),
        "ordering": RECORD_ORDER,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
    }
    _write_meta(path, meta)
    return path, meta


def load_sentiment_dataset(sentiment_path: Path, canonical_path: Path) -> tuple[list[ScoredArticle], dict]:
    """
    Verify the sentiment file hash, verify the canonical dataset is the one it
    was scored from, re-join every record to its canonical article and
    re-validate it (Phase 5A result contract + Phase 5B text-hash match).
    """
    sentiment_path = Path(sentiment_path)
    meta_file = meta_path_for(sentiment_path)
    if not sentiment_path.is_file() or not meta_file.is_file():
        raise FileNotFoundError(f"sentiment dataset or metadata missing: {sentiment_path}")
    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)
    if file_sha256(sentiment_path) != meta.get("sha256"):
        raise HistoricalSentimentError(f"sentiment dataset hash mismatch for {sentiment_path.name}")
    provenance = _require_provenance(meta.get("provenance") or {})

    articles, canonical_meta = load_canonical_dataset(canonical_path)      # hash-verified
    if canonical_meta.get("sha256") != (meta.get("source_dataset") or {}).get("sha256"):
        raise HistoricalSentimentError("canonical dataset does not match the one this sentiment was scored from")
    by_key = {a.key: a for a in articles}

    with open(sentiment_path, encoding="utf-8") as f:
        records = [SentimentRecord.from_dict(json.loads(line)) for line in f if line.strip()]
    if len(records) != meta.get("rows"):
        raise HistoricalSentimentError(f"sentiment dataset has {len(records)} rows, metadata says {meta.get('rows')}")
    check_consistent_provenance((r.sentiment for r in records), provenance)

    scored = []
    for r in records:
        article = by_key.get(r.key)
        if article is None:
            raise HistoricalSentimentError(f"sentiment record {r.key} has no canonical article")
        if (article.information_available_at, article.symbols) != (r.information_available_at, r.symbols):
            raise HistoricalSentimentError(f"sentiment record {r.key} does not match its canonical article")
        try:
            scored.append(ScoredArticle(article, r.sentiment))
        except SentimentFeatureError as e:
            raise HistoricalSentimentError(str(e)) from None
    return prepare_scored_articles(scored), meta


# ==========================================
# Daily dataset
# ==========================================


def trading_dates_from_bars(bars: pd.DataFrame, start: date | None = None, end: date | None = None) -> list[date]:
    """Completed-bar dates with start <= date < end (either bound optional)."""
    dates = sorted(bars["Date"].dt.date)
    return [d for d in dates if (start is None or d >= start) and (end is None or d < end)]


def build_daily_rows(scored: list[ScoredArticle], trading_dates: Iterable, calendar: TradingCalendar,
                     symbols: Iterable[str]) -> list[SentimentFeatureRow]:
    """Phase 5B aggregation, unchanged (information_available_at <= D 16:30 New York)."""
    return generate_sentiment_features(scored, trading_dates, calendar, symbols)


def serialize_daily_rows(rows: Iterable[SentimentFeatureRow]) -> bytes:
    lines = []
    for row in sorted(rows, key=lambda r: (r.symbol, r.trading_date)):
        d = row.to_dict()
        d["trading_date"] = row.trading_date.isoformat()
        d["prediction_timestamp"] = row.prediction_timestamp.isoformat()      # aware UTC, "+00:00"
        lines.append(_dumps(d) + "\n")
    return "".join(lines).encode("utf-8")


def save_daily_dataset(rows: list[SentimentFeatureRow], sentiment_path: Path, sentiment_meta: dict,
                       calendar_source: dict, symbols: list[str], output_dir: Path = SENTIMENT_DIR,
                       generated_at: datetime | None = None) -> tuple[Path, dict]:
    payload = serialize_daily_rows(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / Path(sentiment_path).name.replace(".sentiment.jsonl", ".daily.jsonl")
    _write_once(path, payload)

    dates = sorted({r.trading_date for r in rows})
    meta = {
        "dataset_version": DAILY_DATASET_VERSION,
        "sentiment_feature_version": SENTIMENT_FEATURE_VERSION,
        "eligibility_rule": "information_available_at <= prediction_timestamp (D 16:30 America/New_York)",
        "semantics": "cumulative eligible news as of prediction_timestamp; zero-news rows are all 0",
        "source_sentiment": {"file": Path(sentiment_path).name, "sha256": sentiment_meta.get("sha256")},
        "source_dataset": sentiment_meta.get("source_dataset"),
        "provenance": sentiment_meta.get("provenance"),
        "calendar_source": calendar_source,
        "symbols": list(symbols),
        "first_trading_date": dates[0].isoformat() if dates else None,
        "last_trading_date": dates[-1].isoformat() if dates else None,
        "rows": len(rows),
        "rows_with_news": sum(1 for r in rows if r.news_count > 0),
        "ordering": DAILY_ORDER,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
    }
    _write_meta(path, meta)
    return path, meta


# ==========================================
# End-to-end
# ==========================================


def run_historical_sentiment(canonical_path: Path, bars: pd.DataFrame, service: FinBertSentimentService, *,
                             calendar_source: dict, symbols: list[str] | None = None,
                             start: date | None = None, end: date | None = None,
                             output_dir: Path = SENTIMENT_DIR, checkpoint: Path | None = None) -> dict:
    """
    Score -> persist -> RELOAD (full re-validation) -> aggregate -> persist.
    Daily rows are built from the reloaded data, so what is saved is exactly
    what a later reader would get. `checkpoint` makes scoring resumable.
    """
    articles, canonical_meta = load_canonical_dataset(canonical_path)
    symbols = symbols or canonical_meta.get("symbols")
    if not symbols:
        raise HistoricalSentimentError("no symbols given and none recorded in the canonical dataset metadata")
    # default row range: the news interval's own (UTC) calendar dates, start inclusive, end exclusive
    if start is None and canonical_meta.get("start"):
        start = parse_timestamp(canonical_meta["start"], "start").date()
    if end is None and canonical_meta.get("end"):
        end = parse_timestamp(canonical_meta["end"], "end").date()

    scored, stats = score_canonical_articles(articles, service, checkpoint=checkpoint)
    sentiment_path, sentiment_meta = save_sentiment_dataset(
        scored, stats, backend_provenance(service), canonical_path, canonical_meta, output_dir)

    reloaded, _ = load_sentiment_dataset(sentiment_path, canonical_path)
    calendar = TradingCalendar.from_bars(bars)
    rows = build_daily_rows(reloaded, trading_dates_from_bars(bars, start, end), calendar, symbols)
    daily_path, daily_meta = save_daily_dataset(rows, sentiment_path, sentiment_meta, calendar_source,
                                                sorted({s.upper() for s in symbols}), output_dir)
    return {"sentiment_path": sentiment_path, "sentiment_meta": sentiment_meta,
            "daily_path": daily_path, "daily_meta": daily_meta}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Historical FinBERT scoring -> daily sentiment dataset")
    parser.add_argument("--news", type=Path, required=True, help="canonical news dataset (.jsonl, Phase 4C)")
    parser.add_argument("--market-snapshot", type=Path, required=True,
                        help="Phase 2 raw market snapshot (.csv) defining trading sessions")
    parser.add_argument("--output-dir", type=Path, default=SENTIMENT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--symbols", nargs="+", default=None, help="default: symbols of the news dataset")
    parser.add_argument("--start", type=date.fromisoformat, default=None, help="first trading date (inclusive)")
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="last trading date (exclusive)")
    parser.add_argument("--revision", default=PINNED_FINBERT_REVISION,
                        help="Hugging Face model revision (default: the project's pinned FinBERT commit)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="resumable scoring: append results here and reuse them on rerun")
    args = parser.parse_args(argv)

    bars, snapshot_meta = load_raw_snapshot(args.market_snapshot)          # hash-verified
    service = FinBertSentimentService.load(device="cpu", batch_size=args.batch_size, revision=args.revision)
    out = run_historical_sentiment(
        args.news, bars, service, symbols=args.symbols, start=args.start, end=args.end,
        output_dir=args.output_dir, checkpoint=args.checkpoint,
        calendar_source={"file": args.market_snapshot.name, "sha256": snapshot_meta["sha256"]})

    sm, dm = out["sentiment_meta"], out["daily_meta"]
    print(f"Sentiment : {out['sentiment_path']}")
    print(f"  articles: input {sm['n_input_articles']}, scored {sm['n_scored_articles']}, "
          f"duplicates collapsed {sm['n_duplicates_collapsed']}")
    print(f"  model   : {sm['provenance']['model_name']} @ {sm['provenance']['model_revision']}")
    print(f"Daily     : {out['daily_path']}")
    print(f"  rows    : {dm['rows']} ({dm['rows_with_news']} with news), "
          f"{dm['first_trading_date']} -> {dm['last_trading_date']}")
    print(f"  sha256  : sentiment {sm['sha256'][:16]}...  daily {dm['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
