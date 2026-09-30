"""
Historical news ingestion (Phase 4A).

    NewsProvider.fetch_historical(query)      canonical, deduplicated articles
            |
            v
    raw news snapshot   data/raw/news/<provider>_<SYMBOLS>_<start>_<end>_<UTC stamp>.jsonl
    + metadata          ...same name .meta.json (provenance, sha256)

Reuses the Phase 2 raw-snapshot pattern (training/build_dataset.py):
sha256 of the stored file, a .meta.json next to it, verification on
load, never overwriting an existing snapshot.

Snapshots contain canonical records only (provider fields are kept in
provider_metadata) and never credentials. data/raw/news/ is git-ignored:
full-text provider content is large and licensed.

Usage (from the project root; needs ALPACA_API_KEY / ALPACA_API_SECRET in .env):

    python -m services.news_service --symbols AAPL --start 2017-01-03 --end 2017-01-10

Date-only --start/--end values mean 00:00 UTC.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.news_provider import NewsFetchResult, NewsProvider, NewsQuery  # noqa: E402
from services.news_schema import NEWS_SCHEMA_VERSION, NewsArticle, NewsValidationError  # noqa: E402
from training.build_dataset import file_sha256, meta_path_for  # noqa: E402

RAW_NEWS_DIR = PROJECT_ROOT / "data" / "raw" / "news"


def _stamp(iso_timestamp: str) -> str:
    return datetime.fromisoformat(iso_timestamp).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sort_key(article: NewsArticle):
    return (article.created_at, article.provider, article.provider_article_id)


def serialize_article(article: NewsArticle) -> str:
    """Stable one-line JSON form used by snapshots and the canonical dataset."""
    return json.dumps(article.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def save_news_snapshot(result: NewsFetchResult, query: NewsQuery, raw_dir: Path = RAW_NEWS_DIR) -> Path:
    """
    Write articles as JSON Lines, ordered by (created_at, provider,
    provider_article_id) so the file does not depend on the request's sort
    order, plus a .meta.json with provenance and sha256.
    """
    provenance = result.provenance
    provider = provenance["provider"]

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    name = (f"{provider}_{'-'.join(query.symbols)}_{query.start:%Y%m%d}_{query.end:%Y%m%d}_"
            f"{_stamp(provenance['fetch_finished_at'])}.jsonl")
    path = raw_dir / name
    if path.exists():
        raise FileExistsError(f"News snapshot already exists: {path}")

    articles = sorted(result.articles, key=_sort_key)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for article in articles:
            f.write(serialize_article(article))
            f.write("\n")

    meta = {
        **provenance,
        "schema_version": NEWS_SCHEMA_VERSION,
        "file": path.name,
        "rows": len(articles),
        "first_created_at": articles[0].created_at.isoformat() if articles else None,
        "last_created_at": articles[-1].created_at.isoformat() if articles else None,
        "sha256": file_sha256(path),
    }
    with open(meta_path_for(path), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def load_news_snapshot(path: Path) -> tuple[list[NewsArticle], dict]:
    """Verify the snapshot against its metadata, then parse canonical articles."""
    path = Path(path)
    meta_path = meta_path_for(path)
    if not path.is_file():
        raise FileNotFoundError(f"News snapshot not found: {path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"News snapshot metadata not found: {meta_path}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    actual = file_sha256(path)
    if actual != meta.get("sha256"):
        raise NewsValidationError(f"News snapshot hash mismatch for {path.name}")

    with open(path, encoding="utf-8") as f:
        articles = [NewsArticle.from_dict(json.loads(line)) for line in f if line.strip()]
    if len(articles) != meta.get("rows"):
        raise NewsValidationError(f"News snapshot has {len(articles)} rows, metadata says {meta.get('rows')}")
    return articles, meta


def ingest_historical_news(provider: NewsProvider, query: NewsQuery, *, raw_dir: Path = RAW_NEWS_DIR,
                           max_pages: int | None = None) -> Path:
    """Fetch through `provider` and store a verified raw snapshot."""
    result = provider.fetch_historical(query, max_pages=max_pages)
    return save_news_snapshot(result, query, raw_dir)


def parse_cli_datetime(value: str) -> datetime:
    """'YYYY-MM-DD' -> 00:00 UTC; full ISO-8601 must include an offset."""
    if len(value) == 10:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(f"timestamp needs a timezone offset: {value}")
    return parsed


def main(argv: list[str] | None = None) -> int:
    from services.alpaca_news_provider import AlpacaNewsProvider  # network + credentials

    parser = argparse.ArgumentParser(description="Ingest historical news into a raw snapshot")
    parser.add_argument("--symbols", nargs="+", default=["AAPL"])
    parser.add_argument("--start", type=parse_cli_datetime, required=True)
    parser.add_argument("--end", type=parse_cli_datetime, required=True)
    parser.add_argument("--sort", choices=["asc", "desc"], default="asc")
    parser.add_argument("--no-content", action="store_true", help="request headlines/summaries only")
    parser.add_argument("--page-size", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args(argv)

    query = NewsQuery(symbols=tuple(args.symbols), start=args.start, end=args.end, sort=args.sort,
                      include_content=not args.no_content, page_size=args.page_size)
    provider = AlpacaNewsProvider.from_env()
    path = ingest_historical_news(provider, query, max_pages=args.max_pages)
    _, meta = load_news_snapshot(path)

    print(f"Snapshot : {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")
    print(f"Articles : {meta['rows']} (raw {meta['n_raw_articles']}, duplicates removed "
          f"{meta['n_duplicates_removed']}, conflicting {meta['n_conflicting_duplicates']})")
    print(f"Pages    : {meta['pages']}  |  created {meta['first_created_at']} -> {meta['last_created_at']}")
    print(f"Outside window: {meta['n_created_outside_window']}  |  "
          f"without requested symbol: {meta['n_without_requested_symbol']}")
    print(f"sha256   : {meta['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
