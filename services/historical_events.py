"""
Historical event classification and daily event dataset (Phase 6).

    5C sentiment dataset  (load_sentiment_dataset: verifies sentiment + canonical
                           hashes, re-joins records to canonical articles)
        -> classify_scored_articles   EventClassifier on text_v1 (same text FinBERT scored)
        -> save_event_dataset         <base>.events-<classifier version>.jsonl + .meta.json
        -> load_event_dataset         re-verified, re-joined, optionally RE-CLASSIFIED
                                      to prove the stored labels are reproducible
        -> generate_event_features    event_features_v1 (services/event_features.py)
        -> save_daily_events          <base>.events-<version>.daily.jsonl + .meta.json

No FinBERT re-run: event impact is read from the stored 5C sentiment
(POSITIVE/NEGATIVE/NEUTRAL from the FinBERT label, impact score =
sentiment_score). No new external API, no model download.

Article records hold identity, timestamps, input_text_hash, event fields
and classifier provenance - never article text. The original canonical and
sentiment datasets are never modified.

Leakage: classification is a property of text; when an event may be used
is decided only by information_available_at <= prediction_timestamp
(Phase 4B), via event_features.

Event labels here are CLASSIFIER OUTPUT, not ground truth.

Usage (from the project root; no API keys):

    python -m services.historical_events ^
        --news data/processed/news/<run_id>.jsonl ^
        --sentiment data/processed/sentiment/<run_id>.finbert-<rev>.sentiment.jsonl ^
        --market-snapshot data/raw/stocks/<snapshot>.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.event_classifier import EventClassifier, EventPrediction, KeywordEventClassifier  # noqa: E402
from services.event_features import (  # noqa: E402
    EVENT_FEATURE_VERSION,
    EVENT_OUTPUT_COLUMNS,
    RECENT_SESSIONS,
    ClassifiedArticle,
    EventFeatureError,
    generate_event_features,
    prepare_classified,
)
from services.event_taxonomy import (  # noqa: E402
    EVENT_IMPACTS,
    EVENT_TYPES,
    TAXONOMY_VERSION,
    EventTaxonomyError,
    validate_event_impact,
)
from services.finbert_sentiment import build_sentiment_text  # noqa: E402
from services.historical_sentiment import (  # noqa: E402  (shared write-once / metadata helpers)
    _dumps,
    _write_meta,
    _write_once,
    load_sentiment_dataset,
    trading_dates_from_bars,
)
from services.market_calendar_service import TradingCalendar  # noqa: E402
from services.news_dataset import load_canonical_dataset  # noqa: E402
from services.news_schema import parse_timestamp  # noqa: E402
from services.sentiment_features import ScoredArticle, prepare_scored_articles  # noqa: E402
from training.build_dataset import file_sha256, load_raw_snapshot, meta_path_for  # noqa: E402

EVENTS_DIR = PROJECT_ROOT / "data" / "processed" / "events"
EVENT_RECORD_VERSION = "event_records_v1"
DAILY_EVENTS_VERSION = "daily_events_v1"
RECORD_ORDER = "information_available_at, provider, provider_article_id"
IMPACT_RULE = "event_impact = FinBERT label (positive/negative/neutral); event_impact_score = sentiment_score"
LOW_CONFIDENCE = 0.5

RECORD_FIELDS = ("record_version", "provider", "provider_article_id", "symbols", "published_at",
                 "information_available_at", "input_text_hash", "event_type", "event_confidence",
                 "matched_rules", "event_impact", "event_impact_score", "classifier_name",
                 "classifier_version", "taxonomy_version")


class HistoricalEventError(ValueError):
    """Invalid input, provenance or stored event data."""


# ==========================================
# Classification
# ==========================================


def classify_scored_articles(scored: Iterable[ScoredArticle], classifier: EventClassifier) -> list[ClassifiedArticle]:
    """Classify each unique scored article once, on its text_v1 text. Deterministic order."""
    ordered = prepare_scored_articles(scored)
    texts = [build_sentiment_text(s.article) for s in ordered]
    predictions = classifier.classify_texts(texts)
    if len(predictions) != len(ordered):
        raise HistoricalEventError("classifier returned a different number of predictions")
    return prepare_classified(ClassifiedArticle(s, p) for s, p in zip(ordered, predictions))


def classifier_provenance(classifier: EventClassifier) -> dict:
    return {"classifier_name": classifier.name, "classifier_version": classifier.version,
            "taxonomy_version": TAXONOMY_VERSION}


# ==========================================
# Records
# ==========================================


def record_dict(c: ClassifiedArticle, provenance: dict) -> dict:
    a = c.article
    return {
        "record_version": EVENT_RECORD_VERSION,
        "provider": a.provider,
        "provider_article_id": a.provider_article_id,
        "symbols": list(a.symbols),
        "published_at": a.created_at.isoformat(),
        "information_available_at": a.information_available_at.isoformat(),
        "input_text_hash": c.scored.sentiment.input_text_hash,
        "event_type": c.event_type,
        "event_confidence": c.event_confidence,
        "matched_rules": list(c.prediction.matched_rules),
        "event_impact": c.event_impact,
        "event_impact_score": c.event_impact_score,
        **provenance,
    }


def serialize_event_records(classified: Iterable[ClassifiedArticle], provenance: dict) -> bytes:
    return "".join(_dumps(record_dict(c, provenance)) + "\n" for c in classified).encode("utf-8")


def data_quality_report(classified: list[ClassifiedArticle], n_input: int) -> dict:
    events = [c for c in classified if c.is_event]
    conf = [c.event_confidence for c in events]
    types = Counter(c.event_type for c in classified)
    symbols = Counter(s for c in classified for s in c.article.symbols)
    avail = [c.article.information_available_at for c in classified]
    return {
        "n_input_articles": n_input,
        "n_articles": len(classified),
        "n_duplicates_collapsed": n_input - len(classified),
        "n_events": len(events),
        "n_other": types.get("OTHER", 0),
        "null_or_invalid_event_types": 0,          # enforced by EventPrediction validation
        "event_type_distribution": {t: types.get(t, 0) for t in EVENT_TYPES},
        "event_impact_distribution": {i: sum(1 for c in events if c.event_impact == i) for i in EVENT_IMPACTS},
        "event_confidence": {
            "min": min(conf) if conf else None, "mean": sum(conf) / len(conf) if conf else None,
            "max": max(conf) if conf else None,
            "bins": {"(0,0.5)": sum(1 for x in conf if x < 0.5), "[0.5,0.75)": sum(1 for x in conf if 0.5 <= x < 0.75),
                     "[0.75,1)": sum(1 for x in conf if 0.75 <= x < 1.0), "1.0": sum(1 for x in conf if x == 1.0)},
            "n_low_confidence_events": sum(1 for x in conf if x < LOW_CONFIDENCE),
            "meaning": "share of matched rule evidence won by the chosen type - not a probability",
        },
        "symbol_distribution": dict(sorted(symbols.items())),
        "information_available_at_range": [min(avail).isoformat(), max(avail).isoformat()] if avail else None,
    }


def event_output_path(sentiment_path: Path, classifier: EventClassifier, output_dir: Path = EVENTS_DIR) -> Path:
    name = Path(sentiment_path).name
    if not name.endswith(".sentiment.jsonl"):
        raise HistoricalEventError(f"not a Phase 5C sentiment dataset: {name}")
    return Path(output_dir) / name.replace(".sentiment.jsonl", f".events-{classifier.version}.jsonl")


def save_event_dataset(classified: list[ClassifiedArticle], classifier: EventClassifier, sentiment_path: Path,
                       sentiment_meta: dict, n_input: int, output_dir: Path = EVENTS_DIR,
                       generated_at: datetime | None = None) -> tuple[Path, dict]:
    ordered = prepare_classified(classified)
    provenance = classifier_provenance(classifier)
    payload = serialize_event_records(ordered, provenance)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = event_output_path(sentiment_path, classifier, output_dir)
    _write_once(path, payload)

    meta = {
        "record_version": EVENT_RECORD_VERSION,
        "classifier": provenance,
        "impact_source": {"file": Path(sentiment_path).name, "sha256": sentiment_meta.get("sha256"),
                          "rule": IMPACT_RULE, "sentiment_provenance": sentiment_meta.get("provenance")},
        "source_dataset": sentiment_meta.get("source_dataset"),
        "labels_are_ground_truth": False,
        "data_quality": data_quality_report(ordered, n_input),
        "rows": len(ordered),
        "ordering": RECORD_ORDER,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
    }
    _write_meta(path, meta)
    return path, meta


def load_event_dataset(event_path: Path, sentiment_path: Path, canonical_path: Path, *,
                       verify_with: EventClassifier | None = None) -> tuple[list[ClassifiedArticle], dict]:
    """
    Verify hashes and lineage, re-join every record to its scored article and
    validate it. With `verify_with`, the texts are re-classified and every
    stored prediction must match exactly (reproducibility check).
    """
    event_path = Path(event_path)
    meta_file = meta_path_for(event_path)
    if not event_path.is_file() or not meta_file.is_file():
        raise FileNotFoundError(f"event dataset or metadata missing: {event_path}")
    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)
    if file_sha256(event_path) != meta.get("sha256"):
        raise HistoricalEventError(f"event dataset hash mismatch for {event_path.name}")
    provenance = meta.get("classifier") or {}
    if set(provenance) != {"classifier_name", "classifier_version", "taxonomy_version"} or not all(provenance.values()):
        raise HistoricalEventError("missing classifier provenance in event metadata")
    if provenance["taxonomy_version"] != TAXONOMY_VERSION:
        raise HistoricalEventError(f"event dataset uses {provenance['taxonomy_version']}, code is {TAXONOMY_VERSION}")

    scored, sentiment_meta = load_sentiment_dataset(sentiment_path, canonical_path)     # verifies both hashes
    if sentiment_meta.get("sha256") != (meta.get("impact_source") or {}).get("sha256"):
        raise HistoricalEventError("sentiment dataset does not match the one these events were built from")
    by_key = {s.article.key: s for s in scored}

    classified = []
    with open(event_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            classified.append(_record_to_classified(rec, by_key, provenance))
    if len(classified) != meta.get("rows"):
        raise HistoricalEventError(f"event dataset has {len(classified)} rows, metadata says {meta.get('rows')}")
    classified = prepare_classified(classified)

    if verify_with is not None:
        if classifier_provenance(verify_with) != provenance:
            raise HistoricalEventError("verification classifier differs from the one recorded in the metadata")
        fresh = classify_scored_articles([c.scored for c in classified], verify_with)
        if [c.prediction for c in fresh] != [c.prediction for c in classified]:
            raise HistoricalEventError("re-classification does not reproduce the stored event labels")
    return classified, meta


def _record_to_classified(rec: dict, by_key: dict, provenance: dict) -> ClassifiedArticle:
    missing = [k for k in RECORD_FIELDS if k not in rec or rec[k] is None]
    if missing:
        raise HistoricalEventError(f"event record is missing {missing}")
    key = (rec["provider"], str(rec["provider_article_id"]))
    if rec["record_version"] != EVENT_RECORD_VERSION:
        raise HistoricalEventError(f"unsupported record_version {rec['record_version']!r}")
    if {k: rec[k] for k in provenance} != provenance:
        raise HistoricalEventError(f"event record {key} has different classifier provenance")
    scored = by_key.get(key)
    if scored is None:
        raise HistoricalEventError(f"event record {key} has no scored canonical article")
    a = scored.article
    if (tuple(rec["symbols"]), parse_timestamp(rec["published_at"], "published_at"),
            parse_timestamp(rec["information_available_at"], "information_available_at"),
            rec["input_text_hash"]) != (a.symbols, a.created_at, a.information_available_at,
                                        scored.sentiment.input_text_hash):
        raise HistoricalEventError(f"event record {key} does not match its article / sentiment")
    try:
        prediction = EventPrediction(rec["event_type"], rec["event_confidence"], tuple(rec["matched_rules"]))
        validate_event_impact(rec["event_impact"])
    except (EventTaxonomyError, ValueError) as e:
        raise HistoricalEventError(f"invalid event record {key}: {e}") from None
    c = ClassifiedArticle(scored, prediction)
    if (rec["event_impact"], rec["event_impact_score"]) != (c.event_impact, c.event_impact_score):
        raise HistoricalEventError(f"event record {key} impact does not match its FinBERT sentiment")
    return c


# ==========================================
# Daily dataset
# ==========================================


def serialize_daily_events(rows: Iterable[dict]) -> bytes:
    lines = []
    for row in sorted(rows, key=lambda r: (r["symbol"], r["trading_date"])):
        out = dict(row)
        out["trading_date"] = row["trading_date"].isoformat()
        out["prediction_timestamp"] = row["prediction_timestamp"].isoformat()
        lines.append(_dumps(out) + "\n")
    return "".join(lines).encode("utf-8")


def save_daily_events(rows: list[dict], event_path: Path, event_meta: dict, calendar_source: dict,
                      symbols: list[str], output_dir: Path = EVENTS_DIR,
                      generated_at: datetime | None = None) -> tuple[Path, dict]:
    payload = serialize_daily_events(rows)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / Path(event_path).name.replace(".jsonl", ".daily.jsonl")
    _write_once(path, payload)
    dates = sorted({r["trading_date"] for r in rows})
    meta = {
        "dataset_version": DAILY_EVENTS_VERSION,
        "event_feature_version": EVENT_FEATURE_VERSION,
        "columns": list(EVENT_OUTPUT_COLUMNS),
        "eligibility_rule": "information_available_at <= prediction_timestamp (D 16:30 America/New_York)",
        "window": ("ts(previous session) < information_available_at <= ts(D); "
                   f"recent_event_count over the last {RECENT_SESSIONS} sessions"),
        "zero_convention": "counts/scores 0, dominant_event_type NONE when no events",
        "source_events": {"file": Path(event_path).name, "sha256": event_meta.get("sha256")},
        "classifier": event_meta.get("classifier"),
        "impact_source": event_meta.get("impact_source"),
        "source_dataset": event_meta.get("source_dataset"),
        "calendar_source": calendar_source,
        "symbols": list(symbols),
        "first_trading_date": dates[0].isoformat() if dates else None,
        "last_trading_date": dates[-1].isoformat() if dates else None,
        "rows": len(rows),
        "rows_with_events": sum(1 for r in rows if r["event_count"] > 0),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
    }
    _write_meta(path, meta)
    return path, meta


# ==========================================
# End-to-end
# ==========================================


def run_historical_events(canonical_path: Path, sentiment_path: Path, bars: pd.DataFrame,
                          classifier: EventClassifier | None = None, *, calendar_source: dict,
                          symbols: list[str] | None = None, start: date | None = None, end: date | None = None,
                          output_dir: Path = EVENTS_DIR) -> dict:
    """Classify -> persist -> reload + re-classify (must reproduce) -> aggregate -> persist."""
    classifier = classifier or KeywordEventClassifier()
    _, canonical_meta = load_canonical_dataset(canonical_path)
    scored, sentiment_meta = load_sentiment_dataset(sentiment_path, canonical_path)

    symbols = symbols or canonical_meta.get("symbols")
    if not symbols:
        raise HistoricalEventError("no symbols given and none recorded in the canonical dataset metadata")
    if start is None and canonical_meta.get("start"):
        start = parse_timestamp(canonical_meta["start"], "start").date()
    if end is None and canonical_meta.get("end"):
        end = parse_timestamp(canonical_meta["end"], "end").date()

    started = time.perf_counter()
    classified = classify_scored_articles(scored, classifier)
    classify_seconds = time.perf_counter() - started

    event_path, event_meta = save_event_dataset(classified, classifier, sentiment_path, sentiment_meta,
                                                n_input=len(scored), output_dir=output_dir)
    reloaded, _ = load_event_dataset(event_path, sentiment_path, canonical_path, verify_with=classifier)

    calendar = TradingCalendar.from_bars(bars)
    try:
        rows = generate_event_features(reloaded, trading_dates_from_bars(bars, start, end), calendar, symbols)
    except EventFeatureError as e:
        raise HistoricalEventError(str(e)) from None
    daily_path, daily_meta = save_daily_events(rows, event_path, event_meta, calendar_source,
                                               sorted({s.upper() for s in symbols}), output_dir)
    return {"event_path": event_path, "event_meta": event_meta, "daily_path": daily_path,
            "daily_meta": daily_meta, "classify_seconds": classify_seconds}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Historical event classification -> daily event dataset")
    parser.add_argument("--news", type=Path, required=True, help="canonical news dataset (.jsonl, Phase 4C)")
    parser.add_argument("--sentiment", type=Path, required=True, help="sentiment dataset (.sentiment.jsonl, Phase 5C)")
    parser.add_argument("--market-snapshot", type=Path, required=True, help="Phase 2 raw market snapshot (.csv)")
    parser.add_argument("--output-dir", type=Path, default=EVENTS_DIR)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    args = parser.parse_args(argv)

    bars, snapshot_meta = load_raw_snapshot(args.market_snapshot)
    out = run_historical_events(args.news, args.sentiment, bars, KeywordEventClassifier(),
                                calendar_source={"file": args.market_snapshot.name, "sha256": snapshot_meta["sha256"]},
                                symbols=args.symbols, start=args.start, end=args.end, output_dir=args.output_dir)
    q, dm = out["event_meta"]["data_quality"], out["daily_meta"]
    print(f"Events  : {out['event_path']}")
    print(f"  articles {q['n_articles']} (input {q['n_input_articles']}, duplicates collapsed "
          f"{q['n_duplicates_collapsed']}); events {q['n_events']}, OTHER {q['n_other']}; "
          f"classified in {out['classify_seconds']:.3f}s")
    print("  types   : " + ", ".join(f"{t}={n}" for t, n in q["event_type_distribution"].items() if n))
    print("  impacts : " + ", ".join(f"{i}={n}" for i, n in q["event_impact_distribution"].items()))
    c = q["event_confidence"]
    print(f"  confidence (evidence share, not probability): min {c['min']}, mean {c['mean']}, "
          f"max {c['max']}; low (<{LOW_CONFIDENCE}) {c['n_low_confidence_events']}")
    print(f"Daily   : {out['daily_path']}")
    print(f"  rows {dm['rows']} ({dm['rows_with_events']} with events), "
          f"{dm['first_trading_date']} -> {dm['last_trading_date']}")
    print(f"  sha256  : events {out['event_meta']['sha256'][:16]}...  daily {dm['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
