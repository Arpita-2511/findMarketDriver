"""
Historical news ingestion orchestration (Phase 4C).

    IngestionConfig (symbols, UTC interval, chunking, content mode, paging)
        -> chunk_intervals: consecutive half-open [a, b) windows
        -> for each chunk: provider.fetch_historical(NewsQuery)   (provider paginates)
                           save_news_snapshot(...)                 (Phase 4A, hash + meta)
        -> run manifest  data/raw/news/<run_id>.manifest.json
        -> services/news_dataset.build_canonical_dataset(manifest)

Why chunks: a multi-year query can exceed the provider's page cap, which
aborts the fetch rather than truncating it. Fixed-size chunks keep every
request bounded.

All-or-nothing: the manifest is written only after EVERY chunk succeeded.
A failed run (provider, pagination, validation error) raises and leaves no
manifest, so an incomplete run can never become a dataset. Snapshots of
chunks fetched before the failure stay on disk (hash-verified, unused).

This module never touches HTTP, headers or provider response formats -
only the NewsProvider interface. Credentials never reach the manifest.

Usage (from the project root; ALPACA_API_KEY / ALPACA_API_SECRET in .env):

    python -m services.historical_news_ingestion --symbols AAPL --start 2017-01-01 --end 2017-04-01
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.news_dataset import INTERVAL_RULE, build_canonical_dataset  # noqa: E402
from services.news_provider import SORT_ORDERS, NewsProvider, NewsQuery  # noqa: E402
from services.news_schema import NEWS_SCHEMA_VERSION, NewsValidationError, ensure_utc  # noqa: E402
from services.news_service import (  # noqa: E402
    RAW_NEWS_DIR,
    load_news_snapshot,
    parse_cli_datetime,
    save_news_snapshot,
)
from training.build_dataset import file_sha256, meta_path_for  # noqa: E402

DEFAULT_CHUNK_DAYS = 30
SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def _stamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class IngestionConfig:
    provider: str
    symbols: tuple[str, ...]
    start: datetime
    end: datetime
    chunk_days: int = DEFAULT_CHUNK_DAYS
    include_content: bool = True
    page_size: int | None = None
    sort: str = "asc"
    max_pages: int | None = None

    def __post_init__(self):
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider is required")
        if isinstance(self.symbols, str) or not self.symbols:
            raise ValueError("symbols must be a non-empty sequence of tickers")
        symbols = tuple(sorted({str(s).strip().upper() for s in self.symbols}))
        bad = [s for s in symbols if not SYMBOL_PATTERN.match(s)]
        if bad:
            raise ValueError(f"invalid ticker symbol(s): {bad}")
        object.__setattr__(self, "symbols", symbols)
        try:
            object.__setattr__(self, "start", ensure_utc(self.start, "start"))
            object.__setattr__(self, "end", ensure_utc(self.end, "end"))
        except NewsValidationError as e:
            raise ValueError(str(e)) from e
        if self.start >= self.end:
            raise ValueError("start must be before end")
        if not isinstance(self.chunk_days, int) or self.chunk_days < 1:
            raise ValueError("chunk_days must be a positive integer")
        if self.sort not in SORT_ORDERS:
            raise ValueError(f"sort must be one of {SORT_ORDERS}")

    def describe(self) -> dict:
        """Run parameters for provenance (no credentials exist in the config)."""
        return {
            "provider": self.provider,
            "symbols": list(self.symbols),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "chunk_days": self.chunk_days,
            "include_content": self.include_content,
            "page_size": self.page_size,
            "sort": self.sort,
            "max_pages": self.max_pages,
        }


def chunk_intervals(start: datetime, end: datetime, chunk_days: int) -> list[tuple[datetime, datetime]]:
    """Consecutive half-open windows [a, b) covering [start, end) exactly; the last may be shorter."""
    step = timedelta(days=chunk_days)
    chunks, a = [], start
    while a < end:
        b = min(a + step, end)
        chunks.append((a, b))
        a = b
    return chunks


def find_reusable_snapshot(raw_dir: Path, provider: str, query: NewsQuery) -> tuple[Path, dict] | None:
    """
    Phase 8 resume: an existing chunk snapshot is reusable only if its
    metadata records the SAME provider and request (symbols, exact start/end,
    sort, include_content, page_size). A snapshot without metadata (write
    interrupted) is ignored. Among matches the newest retrieval wins (the UTC
    stamp in the name sorts chronologically). The chosen snapshot is
    hash-verified; corruption raises instead of being silently refetched.
    """
    pattern = f"{provider}_{'-'.join(query.symbols)}_{query.start:%Y%m%d}_{query.end:%Y%m%d}_*.jsonl"
    wanted = query.describe()
    matches = []
    for path in sorted(Path(raw_dir).glob(pattern)):
        meta_file = meta_path_for(path)
        if not meta_file.is_file():
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        request = meta.get("request") or {}
        if meta.get("provider") == provider and all(request.get(k) == v for k, v in wanted.items()):
            matches.append(path)
    if not matches:
        return None
    _, meta = load_news_snapshot(matches[-1])              # sha256 + row count verified
    return matches[-1], meta


def run_historical_ingestion(provider: NewsProvider, config: IngestionConfig, *,
                             raw_dir: Path = RAW_NEWS_DIR,
                             clock: Callable[[], datetime] | None = None,
                             resume: bool = False) -> Path:
    """
    Fetch every chunk through `provider`, store one verified snapshot per
    chunk, then write the run manifest. Any error propagates unchanged
    (NewsProviderError / NewsAuthError / NewsPaginationError /
    NewsValidationError) and no manifest is written.

    resume=True (Phase 8): chunks with a matching, hash-verified snapshot
    from an earlier (possibly failed) run are reused instead of refetched;
    the manifest marks them "reused".
    """
    if provider.name != config.provider:
        raise ValueError(f"provider '{provider.name}' does not match config provider '{config.provider}'")
    clock = clock or (lambda: datetime.now(timezone.utc))
    raw_dir = Path(raw_dir)

    started_at = clock()
    run_id = (f"{config.provider}_{'-'.join(config.symbols)}_{_stamp(config.start)}_"
              f"{_stamp(config.end)}_{_stamp(started_at)}")
    manifest_path = raw_dir / f"{run_id}.manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Ingestion manifest already exists: {manifest_path}")

    chunks = []
    for a, b in chunk_intervals(config.start, config.end, config.chunk_days):
        query = NewsQuery(symbols=config.symbols, start=a, end=b, sort=config.sort,
                          include_content=config.include_content, page_size=config.page_size)
        reused = find_reusable_snapshot(raw_dir, config.provider, query) if resume else None
        if reused is not None:
            snapshot, p = reused
            rows = p["rows"]
        else:
            result = provider.fetch_historical(query, max_pages=config.max_pages)
            snapshot = save_news_snapshot(result, query, raw_dir)
            p, rows = result.provenance, len(result.articles)
        chunks.append({
            "start": a.isoformat(),
            "end": b.isoformat(),
            "snapshot": snapshot.name,
            "sha256": file_sha256(snapshot),
            "rows": rows,
            "pages": p.get("pages"),
            "n_raw_articles": p.get("n_raw_articles"),
            "n_duplicates_removed": p.get("n_duplicates_removed"),
            "n_conflicting_duplicates": p.get("n_conflicting_duplicates"),
            "reused": reused is not None,
        })

    manifest = {
        "run_id": run_id,
        "schema_version": NEWS_SCHEMA_VERSION,
        **config.describe(),
        "interval_rule": INTERVAL_RULE,
        "started_at": started_at.isoformat(),
        "finished_at": clock().isoformat(),
        "chunks": chunks,
        "totals": {
            "chunks": len(chunks),
            "pages": sum(c["pages"] or 0 for c in chunks),
            "n_raw_articles": sum(c["n_raw_articles"] or 0 for c in chunks),
            "n_snapshot_rows": sum(c["rows"] for c in chunks),
            "n_provider_duplicates_removed": sum(c["n_duplicates_removed"] or 0 for c in chunks),
            "n_reused_chunks": sum(1 for c in chunks if c["reused"]),
        },
    }
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest_path


# ==========================================
# CLI
# ==========================================


def _alpaca_factory() -> NewsProvider:
    from services.alpaca_news_provider import AlpacaNewsProvider
    return AlpacaNewsProvider.from_env()


PROVIDER_FACTORIES: dict[str, Callable[[], NewsProvider]] = {"alpaca": _alpaca_factory}


def create_provider(name: str) -> NewsProvider:
    try:
        return PROVIDER_FACTORIES[name]()
    except KeyError:
        raise ValueError(f"unknown news provider '{name}'; available: {sorted(PROVIDER_FACTORIES)}") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Historical news ingestion -> canonical dataset")
    parser.add_argument("--provider", default="alpaca", choices=sorted(PROVIDER_FACTORIES))
    parser.add_argument("--symbols", nargs="+", default=["AAPL"])
    parser.add_argument("--start", type=parse_cli_datetime, required=True)
    parser.add_argument("--end", type=parse_cli_datetime, required=True)
    parser.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS)
    parser.add_argument("--no-content", action="store_true", help="headlines/summaries only")
    parser.add_argument("--page-size", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None, help="per chunk")
    parser.add_argument("--resume", action="store_true",
                        help="reuse matching, hash-verified chunk snapshots from earlier runs")
    args = parser.parse_args(argv)

    config = IngestionConfig(provider=args.provider, symbols=tuple(args.symbols), start=args.start,
                             end=args.end, chunk_days=args.chunk_days,
                             include_content=not args.no_content, page_size=args.page_size,
                             max_pages=args.max_pages)
    manifest_path = run_historical_ingestion(create_provider(config.provider), config, resume=args.resume)
    dataset_path, meta = build_canonical_dataset(manifest_path)

    rel = lambda p: p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p  # noqa: E731
    print(f"Manifest : {rel(manifest_path)}")
    print(f"Dataset  : {rel(dataset_path)}")
    print(f"Chunks   : {len(meta['source_snapshots'])}  |  input records {meta['n_input_records']}")
    print(f"Articles : {meta['rows']}  (duplicates removed {meta['n_duplicates_removed']}, "
          f"conflicting {meta['n_conflicting_duplicates']}, outside interval {meta['n_outside_interval']}, "
          f"without requested symbol {meta['n_without_requested_symbol']})")
    print(f"Created  : {meta['first_created_at']} -> {meta['last_created_at']}")
    print(f"sha256   : {meta['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
