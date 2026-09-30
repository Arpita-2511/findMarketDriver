"""
Canonical historical news dataset (Phase 4C).

    run manifest (services/historical_news_ingestion.py)
        -> raw chunk snapshots (hash-verified, services/news_service.py)
        -> canonicalize: dedupe -> interval filter -> symbol filter -> sort
        -> data/processed/news/<run_id>.jsonl + .meta.json

The dataset is a pure function of the manifest and its snapshots: the
same snapshots always produce byte-identical output, whatever order the
snapshots, pages or articles arrived in.

Rules (ARCHITECTURE.md 11.4):

INTERVAL_RULE  start <= created_at < end. The ingestion interval selects
               which PUBLISHED articles belong to the dataset; half-open
               so consecutive chunks tile without gaps or double counting.
               updated_at / information_available_at may fall after `end`;
               they are kept unchanged. This is NOT prediction eligibility
               - that remains information_available_at <= prediction_timestamp
               (services/news_alignment.py), applied later.

SYMBOL_RULE    keep an article iff its canonical symbols contain at least one
               requested symbol; its symbols are kept as provided (never
               added, removed or inferred from text).

DEDUP_RULE     identity = (provider, provider_article_id). Across snapshots,
               keep the version with the latest (information_available_at,
               fetched_at); remaining ties -> smallest serialized record.
               Order-independent. Keeping the latest revision is also the
               leakage-conservative choice (its availability is latest).
               Headline similarity is never used.

Data under data/processed/news/ contains licensed full text and is
git-ignored, like the raw snapshots.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.news_schema import (  # noqa: E402
    HISTORICAL_AVAILABILITY_RULE,
    NEWS_SCHEMA_VERSION,
    NewsArticle,
    NewsValidationError,
    parse_timestamp,
)
from services.news_service import load_news_snapshot, serialize_article  # noqa: E402
from training.build_dataset import file_sha256, meta_path_for  # noqa: E402

CANONICAL_NEWS_DIR = PROJECT_ROOT / "data" / "processed" / "news"
DATASET_VERSION = "news_dataset_v1"

INTERVAL_RULE = "start <= created_at < end (half-open, publication time)"
SYMBOL_RULE = "keep iff canonical symbols intersect requested symbols; symbols unchanged"
DEDUP_RULE = ("identity (provider, provider_article_id); keep latest "
              "(information_available_at, fetched_at), then smallest serialized record")


# ==========================================
# Canonicalization (pure functions)
# ==========================================


def in_interval(article: NewsArticle, start, end) -> bool:
    return start <= article.created_at < end


def has_requested_symbol(article: NewsArticle, symbols: Iterable[str]) -> bool:
    return bool(set(article.symbols) & {s.strip().upper() for s in symbols})


def _preference(article: NewsArticle):
    return (article.information_available_at, article.fetched_at)


def dedupe_canonical(articles: Iterable[NewsArticle]) -> tuple[list[NewsArticle], int, int]:
    """
    Order-independent deduplication (DEDUP_RULE).
    Returns (unique articles, duplicates removed, conflicting duplicates),
    where "conflicting" means the duplicate's serialized record differed.
    """
    best: dict[tuple[str, str], tuple[NewsArticle, str]] = {}
    duplicates = conflicts = 0
    for article in articles:
        text = serialize_article(article)
        current = best.get(article.key)
        if current is None:
            best[article.key] = (article, text)
            continue
        duplicates += 1
        kept, kept_text = current
        if text != kept_text:
            conflicts += 1
        if _prefer(article, text, kept, kept_text):
            best[article.key] = (article, text)
    return [a for a, _ in best.values()], duplicates, conflicts


def _prefer(candidate: NewsArticle, candidate_text: str, kept: NewsArticle, kept_text: str) -> bool:
    """True if `candidate` should replace `kept` under DEDUP_RULE."""
    if _preference(candidate) != _preference(kept):
        return _preference(candidate) > _preference(kept)
    return candidate_text < kept_text


def sort_key(article: NewsArticle):
    return (article.created_at, article.provider, article.provider_article_id)


def canonicalize(articles: Iterable[NewsArticle], *, symbols: Iterable[str], start, end
                 ) -> tuple[list[NewsArticle], dict]:
    """Dedupe -> interval filter -> symbol filter -> deterministic sort. Returns (articles, stats)."""
    articles = list(articles)
    symbols = [s.strip().upper() for s in symbols]
    unique, n_dup, n_conflict = dedupe_canonical(articles)

    in_window = [a for a in unique if in_interval(a, start, end)]
    matching = [a for a in in_window if has_requested_symbol(a, symbols)]
    result = sorted(matching, key=sort_key)

    stats = {
        "n_input_records": len(articles),
        "n_duplicates_removed": n_dup,
        "n_conflicting_duplicates": n_conflict,
        "n_outside_interval": len(unique) - len(in_window),
        "n_without_requested_symbol": len(in_window) - len(matching),
        "n_articles": len(result),
    }
    return result, stats


def serialize_dataset(articles: Iterable[NewsArticle]) -> bytes:
    """Stable JSON Lines bytes (UTF-8, '\\n'), in the given order."""
    return "".join(serialize_article(a) + "\n" for a in articles).encode("utf-8")


# ==========================================
# Build from a run manifest
# ==========================================


def load_manifest(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Ingestion manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    for key in ("run_id", "provider", "symbols", "start", "end", "chunks"):
        if key not in manifest:
            raise NewsValidationError(f"Ingestion manifest is missing '{key}'")
    return manifest


def load_manifest_articles(manifest_path: Path) -> tuple[list[NewsArticle], dict]:
    """Load every chunk snapshot listed in the manifest, verifying each hash twice
    (against its own .meta.json and against the manifest)."""
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    articles: list[NewsArticle] = []
    for chunk in manifest["chunks"]:
        path = manifest_path.parent / chunk["snapshot"]
        chunk_articles, meta = load_news_snapshot(path)
        if meta["sha256"] != chunk["sha256"]:
            raise NewsValidationError(f"Snapshot {chunk['snapshot']} does not match the manifest hash")
        articles.extend(chunk_articles)
    return articles, manifest


def build_canonical_dataset(manifest_path: Path, output_dir: Path = CANONICAL_NEWS_DIR) -> tuple[Path, dict]:
    """Deterministically build the canonical dataset for one ingestion run."""
    manifest_path = Path(manifest_path)
    articles, manifest = load_manifest_articles(manifest_path)
    start = parse_timestamp(manifest["start"], "start")
    end = parse_timestamp(manifest["end"], "end")

    dataset, stats = canonicalize(articles, symbols=manifest["symbols"], start=start, end=end)
    payload = serialize_dataset(dataset)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{manifest['run_id']}.jsonl"
    path.write_bytes(payload)

    meta = {
        "dataset_version": DATASET_VERSION,
        "schema_version": NEWS_SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "provider": manifest["provider"],
        "symbols": manifest["symbols"],
        "start": manifest["start"],
        "end": manifest["end"],
        "include_content": manifest.get("include_content"),
        "interval_rule": INTERVAL_RULE,
        "symbol_rule": SYMBOL_RULE,
        "dedup_rule": DEDUP_RULE,
        "availability_rule": HISTORICAL_AVAILABILITY_RULE,
        "manifest": {"file": manifest_path.name, "sha256": file_sha256(manifest_path)},
        "source_snapshots": [{"file": c["snapshot"], "sha256": c["sha256"]} for c in manifest["chunks"]],
        **stats,
        "rows": len(dataset),
        "first_created_at": dataset[0].created_at.isoformat() if dataset else None,
        "last_created_at": dataset[-1].created_at.isoformat() if dataset else None,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    with open(meta_path_for(path), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    return path, meta


def load_canonical_dataset(path: Path) -> tuple[list[NewsArticle], dict]:
    """Read a canonical dataset after verifying its sha256."""
    path = Path(path)
    meta_path = meta_path_for(path)
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Canonical news dataset or metadata missing: {path}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if file_sha256(path) != meta.get("sha256"):
        raise NewsValidationError(f"Canonical news dataset hash mismatch for {path.name}")
    with open(path, encoding="utf-8") as f:
        articles = [NewsArticle.from_dict(json.loads(line)) for line in f if line.strip()]
    return articles, meta
