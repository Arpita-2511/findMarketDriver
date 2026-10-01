"""
Backfill audit of a canonical news dataset (Phase 8). Read-only: reports,
never deletes or merges anything.

Checks (on the hash-verified dataset from services/news_dataset.py):
    - duplicate article identities (provider, provider_article_id) - must be 0
    - duplicate source URLs and same (headline, created_at) pairs under
      DIFFERENT ids - reported only; distinct ids are kept (syndicated or
      re-filed stories are legitimate distinct records for the provider)
    - articles per year and calendar months inside the interval with no articles
    - availability after the regular close (16:00 New York) and on weekends,
      i.e. news that can only reach the NEXT session's prediction row
    - revised articles (information_available_at > created_at)
    - summary / content presence (content is not needed by text_v1 when a
      headline exists)

Usage:  python -m services.news_audit --news data/processed/news/<run_id>.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.finbert_sentiment import normalize_whitespace  # noqa: E402
from services.market_calendar_service import REGULAR_SESSION_CLOSE  # noqa: E402
from services.news_alignment import to_exchange_time  # noqa: E402
from services.news_dataset import load_canonical_dataset  # noqa: E402
from services.news_schema import parse_timestamp  # noqa: E402


def _months(start, end) -> list[str]:
    y, m, out = start.year, start.month, []
    while (y, m) < (end.year, end.month) or (y, m) == (end.year, end.month) and end.day > 1:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def audit_canonical_dataset(path: Path) -> dict:
    articles, meta = load_canonical_dataset(path)                 # sha256 verified
    keys = Counter(a.key for a in articles)
    urls = Counter(a.source_url for a in articles if a.source_url)
    same_story = Counter((normalize_whitespace(a.headline).casefold(), a.created_at) for a in articles)
    local = [to_exchange_time(a.information_available_at) for a in articles]
    per_month = Counter(a.created_at.strftime("%Y-%m") for a in articles)

    report = {
        "dataset": Path(path).name,
        "sha256": meta.get("sha256"),
        "articles": len(articles),
        "duplicate_article_ids": sum(n - 1 for n in keys.values() if n > 1),
        "duplicate_source_urls": {"urls": sum(1 for n in urls.values() if n > 1),
                                  "extra_records": sum(n - 1 for n in urls.values() if n > 1)},
        "same_headline_and_created_at_different_ids": sum(n - 1 for n in same_story.values() if n > 1),
        "articles_per_year": dict(sorted(Counter(a.created_at.year for a in articles).items())),
        "created_at_range": ([min(a.created_at for a in articles).isoformat(),
                              max(a.created_at for a in articles).isoformat()] if articles else None),
        "available_after_close_or_weekend": sum(
            1 for t in local if t.weekday() >= 5 or t.timetz().replace(tzinfo=None) > REGULAR_SESSION_CLOSE),
        "revised_after_creation": sum(1 for a in articles if a.information_available_at > a.created_at),
        "with_summary": sum(1 for a in articles if a.summary),
        "with_content": sum(1 for a in articles if a.content),
        "symbols": dict(sorted(Counter(s for a in articles for s in a.symbols).items())),
    }
    if meta.get("start") and meta.get("end"):
        months = _months(parse_timestamp(meta["start"], "start"), parse_timestamp(meta["end"], "end"))
        report["interval"] = [meta["start"], meta["end"]]
        report["months_in_interval"] = len(months)
        report["months_without_articles"] = [m for m in months if per_month.get(m, 0) == 0]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a canonical news dataset (read-only)")
    parser.add_argument("--news", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit_canonical_dataset(args.news), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
