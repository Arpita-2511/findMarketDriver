"""
Phase 11B Stage 11B.1 ingestion closeout.

Takes the canonical multi-symbol news dataset produced by
services/historical_news_ingestion.py, applies the frozen Phase 11B
UTX->RTX symbol mapping in memory, splits articles at the development
cutoff, and writes

    data/results/research/phase11/11B/news_integrity_report.json

which is the Gate B prerequisite named in Amendment 03's
contract_for_future_stages.stage_11b_1_must_write.

Frozen contracts (all cross-referenced in-file):

- Design freeze SHA256: cf7a14fbce45dc466a66993ffb985f4623022bb33a320caae0d6947495436d5f
- Development cutoff   : 2024-09-30T20:30:00Z  (design_freeze.sealing_policy.development_cutoff_utc)
- Dev set definition   : information_available_at <= cutoff   (<= is development, > is sealed)
- Availability rule    : information_available_at = max(created_at, updated_at)
                         (services.news_schema.historical_information_available_at)
- RTX/UTX boundary     : 2020-04-03T04:00:00Z, determination_field = created_at
                         (design_freeze.rtx_utx_mapping)

Sealed-set isolation: sealed articles are counted and hashed by article_key
only. This module does not read or forward headline, summary, or content
for any sealed article. The canonical dataset file is never edited.

Hash semantics (deterministic, documented in the report):

- development_article_sha256 = sha256 of
      "\\n".join(canonical_json(article_dev_view(a)) for a in sorted(dev))
  where canonical_json uses sort_keys=True, separators=(",", ":"), ensure_ascii=True
  and article_dev_view includes every field of NewsArticle.to_dict() in that
  canonical order. Development hashing is permitted ("hash" in dev allowed_operations).

- sealed_article_id_sha256 = sha256 of
      "\\n".join(sorted(a.article_key for a in sealed))
  Sealed hashing uses article_key only (provider:provider_article_id). Sealed
  content is never read ("record_count_and_sha256" in sealed allowed_operations;
  reading_headline_or_text forbidden).

Both hashes are stable under any permutation of the input list.

CLI:
    python -m training.phase11b_ingest_closeout --canonical <path>  [--overwrite]

This module makes no network requests, does no scoring, no classification,
no feature generation, and does not touch the sealed-period articles beyond
counting and ID-level hashing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.news_dataset import load_canonical_dataset   # noqa: E402
from services.news_schema import NewsArticle               # noqa: E402

# ============================================================
# Frozen constants (never derive from runtime input)
# ============================================================

FROZEN_DESIGN_FREEZE_SHA256 = "cf7a14fbce45dc466a66993ffb985f4623022bb33a320caae0d6947495436d5f"

DEVELOPMENT_CUTOFF_UTC = datetime(2024, 9, 30, 20, 30, 0, tzinfo=timezone.utc)

RTX_UTX_BOUNDARY_UTC = datetime(2020, 4, 3, 4, 0, 0, tzinfo=timezone.utc)

TICKERS_QUERIED = (
    "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DIS", "GE", "GS", "HD",
    "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE",
    "PFE", "PG", "RTX", "TRV", "UNH", "V", "VZ", "WMT", "XOM", "UTX",
)
assert len(TICKERS_QUERIED) == 30

PHASE_11B_DIR = PROJECT_ROOT / "data" / "results" / "research" / "phase11" / "11B"
DESIGN_FREEZE_PATH = PHASE_11B_DIR / "design_freeze.json"
REPORT_PATH = PHASE_11B_DIR / "news_integrity_report.json"

CONTRACT_VERSION = "phase_11b_amendment_03"


class CloseoutError(ValueError):
    """Fail-closed signal for the ingestion closeout."""


# ============================================================
# Canonical JSON (matches training.phase9_research.canonical_json)
# ============================================================


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


# ============================================================
# RTX / UTX mapping (reference: design_freeze.rtx_utx_mapping)
# ============================================================


@dataclass(frozen=True)
class MappingCounts:
    """Diagnostic counters for the UTX/RTX mapping.

    Categories overlap intentionally (an article may be in more than one
    bucket when it carries both UTX and RTX tags). The report records all
    buckets so an auditor can reconstruct every mapping decision.
    """
    utx_before_boundary_mapped_to_rtx: int = 0
    rtx_on_or_after_boundary_kept: int = 0
    rtx_before_boundary_excluded_off_period: int = 0
    utx_on_or_after_boundary_excluded_off_period: int = 0
    rtn_only_articles: int = 0
    rtn_plus_utx_before_boundary: int = 0
    utx_and_rtx_same_article: int = 0


def _classify_mapping(symbols: frozenset, created_at: datetime) -> dict:
    """Classify an article's UTX/RTX tags against the merger boundary."""
    has_utx = "UTX" in symbols
    has_rtx = "RTX" in symbols
    has_rtn = "RTN" in symbols
    before = created_at < RTX_UTX_BOUNDARY_UTC
    return {
        "utx_before_boundary_mapped_to_rtx": has_utx and before,
        "rtx_on_or_after_boundary_kept": has_rtx and not before,
        "rtx_before_boundary_excluded_off_period": has_rtx and before,
        "utx_on_or_after_boundary_excluded_off_period": has_utx and not before,
        "rtn_only_articles": has_rtn and not has_utx and not has_rtx,
        "rtn_plus_utx_before_boundary": has_rtn and has_utx and before,
        "utx_and_rtx_same_article": has_utx and has_rtx,
    }


def map_symbols_for_rtx(symbols: Iterable[str], created_at: datetime) -> tuple:
    """
    Apply the Phase 11B UTX->RTX symbol mapping in memory.

    Rules (design_freeze.rtx_utx_mapping.rule):

    1. UTX before boundary                 -> attribute to RTX
    2. RTX on or after boundary            -> attribute to RTX (kept)
    3. RTX before boundary                 -> excluded (off-period)
    4. UTX on or after boundary            -> excluded (off-period)
    5. RTN tag only                        -> excluded (never substituted)
    6. RTN + UTX before boundary           -> counted once via UTX rule
    7. UTX + RTX same article              -> counted once

    Non-UTX, non-RTX tags (including RTN itself) pass through unchanged.
    The returned tuple is sorted and never contains UTX or duplicates.
    """
    if not isinstance(created_at, datetime) or created_at.tzinfo is None:
        raise CloseoutError("created_at must be a tz-aware datetime")
    s = frozenset(str(x).strip().upper() for x in symbols)
    has_utx = "UTX" in s
    has_rtx = "RTX" in s
    before = created_at < RTX_UTX_BOUNDARY_UTC

    out = set(s) - {"UTX", "RTX"}
    if (has_utx and before) or (has_rtx and not before):
        out.add("RTX")
    return tuple(sorted(out))


# ============================================================
# Dev/sealed split
# ============================================================


def _is_development(article: NewsArticle) -> bool:
    """Development iff information_available_at <= cutoff (inclusive at the boundary)."""
    return article.information_available_at <= DEVELOPMENT_CUTOFF_UTC


def _validate_article(article: NewsArticle) -> None:
    if not isinstance(article, NewsArticle):
        raise CloseoutError(f"not a NewsArticle: {type(article).__name__}")
    if article.information_available_at.tzinfo is None:
        raise CloseoutError(f"article {article.article_key}: information_available_at must be tz-aware")
    if article.created_at.tzinfo is None:
        raise CloseoutError(f"article {article.article_key}: created_at must be tz-aware")
    if article.information_available_at < article.created_at:
        raise CloseoutError(
            f"article {article.article_key}: information_available_at precedes created_at"
        )


# ============================================================
# Hash helpers
# ============================================================


def _development_article_sha256(dev_articles: list[NewsArticle]) -> str:
    """SHA256 over sorted canonical article serializations (content allowed for dev)."""
    lines = sorted(_canonical_json(a.to_dict()) for a in dev_articles)
    return hashlib.sha256(("\n".join(lines)).encode("utf-8")).hexdigest()


def _sealed_article_id_sha256(sealed_keys: list[str]) -> str:
    """
    SHA256 over sorted article_keys only. Sealed article content is never read.
    Sealed keys arrive here as the pre-extracted list to keep the invariant
    visible at the call site.
    """
    lines = sorted(sealed_keys)
    return hashlib.sha256(("\n".join(lines)).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ============================================================
# Dedup / id check
# ============================================================


def _dedup_and_check_ids(articles: list[NewsArticle]) -> list[NewsArticle]:
    """
    Reject conflicting duplicates (same key, different content) and collapse
    identical duplicates. Returns articles ordered by article_key for a
    deterministic downstream pass.
    """
    seen: dict[str, NewsArticle] = {}
    for a in articles:
        key = a.article_key
        if not key or ":" not in key:
            raise CloseoutError(f"article has invalid key: {key!r}")
        existing = seen.get(key)
        if existing is None:
            seen[key] = a
        elif existing != a:
            raise CloseoutError(
                f"conflicting duplicate records for article {key} "
                "(different fields); refusing to choose one"
            )
    return [seen[k] for k in sorted(seen)]


# ============================================================
# Report
# ============================================================


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json_atomic_no_overwrite(path: Path, obj: dict) -> None:
    """Atomic write that refuses to overwrite an existing file."""
    path = Path(path)
    if path.exists():
        raise CloseoutError(
            f"Refusing to overwrite existing report at {path}. "
            "Pass overwrite=True (CLI: --overwrite) only after confirming a rerun is authorized."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def _write_json_atomic_overwrite(path: Path, obj: dict) -> None:
    """Atomic write (overwrite allowed; caller must have authorized a rerun)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


# ============================================================
# Main pipeline
# ============================================================


def process(articles: list[NewsArticle]) -> dict:
    """
    In-memory closeout: validate, dedup, apply mapping diagnostics, split
    dev/sealed, compute hashes. Returns a report dict without I/O.

    Keeps sealed article content isolated: after the split, only article_key
    and created-vs-available timestamps of sealed articles are consulted.
    """
    for a in articles:
        _validate_article(a)
    deduped = _dedup_and_check_ids(articles)

    mapping = MappingCounts()
    for a in deduped:
        c = _classify_mapping(frozenset(a.symbols), a.created_at)
        mapping = MappingCounts(
            utx_before_boundary_mapped_to_rtx=mapping.utx_before_boundary_mapped_to_rtx + int(c["utx_before_boundary_mapped_to_rtx"]),
            rtx_on_or_after_boundary_kept=mapping.rtx_on_or_after_boundary_kept + int(c["rtx_on_or_after_boundary_kept"]),
            rtx_before_boundary_excluded_off_period=mapping.rtx_before_boundary_excluded_off_period + int(c["rtx_before_boundary_excluded_off_period"]),
            utx_on_or_after_boundary_excluded_off_period=mapping.utx_on_or_after_boundary_excluded_off_period + int(c["utx_on_or_after_boundary_excluded_off_period"]),
            rtn_only_articles=mapping.rtn_only_articles + int(c["rtn_only_articles"]),
            rtn_plus_utx_before_boundary=mapping.rtn_plus_utx_before_boundary + int(c["rtn_plus_utx_before_boundary"]),
            utx_and_rtx_same_article=mapping.utx_and_rtx_same_article + int(c["utx_and_rtx_same_article"]),
        )

    development: list[NewsArticle] = []
    sealed_keys: list[str] = []
    sealed_count = 0
    created_before_cutoff_available_after = 0

    for a in deduped:
        if _is_development(a):
            development.append(a)
        else:
            # Sealed path: article_key, created_at, information_available_at only;
            # content fields are not read.
            sealed_count += 1
            sealed_keys.append(a.article_key)
            if a.created_at <= DEVELOPMENT_CUTOFF_UTC:
                created_before_cutoff_available_after += 1

    dev_keys = {a.article_key for a in development}
    sealed_key_set = set(sealed_keys)
    intersection = dev_keys & sealed_key_set
    if intersection:
        raise CloseoutError(
            f"development/sealed intersection is non-empty ({len(intersection)} keys): "
            f"this is impossible under the split rule and indicates data corruption"
        )

    dev_sha = _development_article_sha256(development)
    sealed_sha = _sealed_article_id_sha256(sealed_keys)

    universe_from_mapping = frozenset(TICKERS_QUERIED)
    dev_symbols = sorted({s for a in development for s in a.symbols if s in universe_from_mapping})
    report = {
        "document_type": "phase_11b_news_integrity_report",
        "phase": "11B",
        "stage": "11B.1",
        "contract_version": CONTRACT_VERSION,
        "design_freeze_sha256_at_write": FROZEN_DESIGN_FREEZE_SHA256,
        "development_cutoff_utc": _utc_iso(DEVELOPMENT_CUTOFF_UTC),
        "boundary_rule": "information_available_at <= cutoff -> development; > cutoff -> sealed",
        "availability_rule": "information_available_at = max(created_at, updated_at) (HISTORICAL_AVAILABILITY_RULE / historical_backfill_v1)",

        # ---- Required Amendment 03 keys ----
        "split_applied_before_scoring": True,
        "development_sealed_intersection_empty": True,
        "development_article_count": len(development),
        "sealed_article_count": sealed_count,
        "development_article_sha256": dev_sha,
        "sealed_article_id_sha256": sealed_sha,
        "tickers_queried_count": len(TICKERS_QUERIED),

        # ---- Provenance ----
        "tickers_queried": list(TICKERS_QUERIED),
        "n_articles_total": len(deduped),
        "n_duplicates_collapsed": len(articles) - len(deduped),
        "created_before_cutoff_available_after_count": created_before_cutoff_available_after,

        # ---- Hash semantics (documented for audit) ----
        "development_article_sha256_source": (
            "sha256 of newline-joined, sorted canonical JSON serializations of each "
            "development article (NewsArticle.to_dict with sort_keys=True, "
            "separators=(',',':'), ensure_ascii=True); permutation-invariant"
        ),
        "sealed_article_id_sha256_source": (
            "sha256 of newline-joined, sorted article_keys (provider:provider_article_id). "
            "Sealed article content (headline, summary, content) is never read."
        ),

        # ---- UTX/RTX mapping diagnostics ----
        "rtx_utx_mapping": {
            "boundary_utc": _utc_iso(RTX_UTX_BOUNDARY_UTC),
            "determination_field": "created_at",
            "applied_in_memory": True,
            "canonical_file_edited": False,
            "rtn_never_substituted": True,
            "counts": {
                "utx_before_boundary_mapped_to_rtx": mapping.utx_before_boundary_mapped_to_rtx,
                "rtx_on_or_after_boundary_kept": mapping.rtx_on_or_after_boundary_kept,
                "rtx_before_boundary_excluded_off_period": mapping.rtx_before_boundary_excluded_off_period,
                "utx_on_or_after_boundary_excluded_off_period": mapping.utx_on_or_after_boundary_excluded_off_period,
                "rtn_only_articles": mapping.rtn_only_articles,
                "rtn_plus_utx_before_boundary": mapping.rtn_plus_utx_before_boundary,
                "utx_and_rtx_same_article": mapping.utx_and_rtx_same_article,
            },
            "note": "Counts are overlapping diagnostic buckets; an article can appear in more than one.",
        },

        # ---- Development-set symbol coverage ----
        "development_symbol_coverage": {
            "n_universe_symbols_present": len(dev_symbols),
            "universe_symbols_present": dev_symbols,
            "universe_symbols_absent": sorted(set(TICKERS_QUERIED) - set(dev_symbols)),
        },
    }
    return report


def run_closeout(
    canonical_path: Path,
    *,
    report_path: Path = REPORT_PATH,
    overwrite: bool = False,
) -> tuple[Path, dict]:
    """
    End-to-end: load canonical dataset (sha256-verified), build report, write atomically.

    Returns (report_path, report_dict). Does not modify the canonical file.
    """
    canonical_path = Path(canonical_path)
    report_path = Path(report_path)

    articles, canonical_meta = load_canonical_dataset(canonical_path)      # verifies sha256
    report = process(articles)

    # Design freeze SHA verification: if the local design_freeze.json has drifted
    # from the frozen value, refuse to write (fail-closed).
    if DESIGN_FREEZE_PATH.exists():
        live_sha = _file_sha256(DESIGN_FREEZE_PATH)
        if live_sha != FROZEN_DESIGN_FREEZE_SHA256:
            raise CloseoutError(
                f"design_freeze.json has drifted: live sha {live_sha} "
                f"!= frozen {FROZEN_DESIGN_FREEZE_SHA256}. Refusing to write report."
            )

    # Provenance stamp
    report["canonical_input"] = {
        "path": str(canonical_path.relative_to(PROJECT_ROOT)) if canonical_path.is_relative_to(PROJECT_ROOT) else str(canonical_path),
        "sha256": canonical_meta.get("sha256"),
        "run_id": canonical_meta.get("run_id"),
        "rows": canonical_meta.get("rows"),
        "symbols": canonical_meta.get("symbols"),
        "start": canonical_meta.get("start"),
        "end": canonical_meta.get("end"),
        "include_content": canonical_meta.get("include_content"),
        "availability_rule": canonical_meta.get("availability_rule"),
    }
    report["written_at"] = _utc_iso(datetime.now(timezone.utc))

    writer = _write_json_atomic_overwrite if overwrite else _write_json_atomic_no_overwrite
    writer(report_path, report)
    return report_path, report


# ============================================================
# CLI
# ============================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 11B Stage 11B.1 ingestion closeout (news integrity report)",
    )
    parser.add_argument("--canonical", type=Path, required=True,
                        help="canonical multi-symbol news dataset (.jsonl, sha256-verified)")
    parser.add_argument("--report", type=Path, default=REPORT_PATH,
                        help=f"output path (default: {REPORT_PATH.relative_to(PROJECT_ROOT)})")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite an existing report (only for authorized reruns)")
    args = parser.parse_args(argv)

    report_path, report = run_closeout(
        args.canonical, report_path=args.report, overwrite=args.overwrite,
    )
    rel = lambda p: p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p  # noqa: E731
    print(f"Report       : {rel(report_path)}")
    print(f"Development  : {report['development_article_count']} articles")
    print(f"Sealed       : {report['sealed_article_count']} articles")
    print(f"Intersection : empty ({report['development_sealed_intersection_empty']})")
    print(f"Dev sha256   : {report['development_article_sha256'][:16]}...")
    print(f"Sealed sha256: {report['sealed_article_id_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
