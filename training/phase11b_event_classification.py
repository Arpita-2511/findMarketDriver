"""
Phase 11B Stage 11B.2 (second step): development-only rules_v1 event classification.

    integrity report (Gate B)  ->  canonical path, sha256, partition counts and hashes
    11B.2 scoring run record   ->  development score file, its sha256, model provenance
        -> partition_canonical         same streaming split as scoring; sealed records keep
                                       only their key and must reproduce Gate B exactly
        -> development score records   joined to development articles only (article set
                                       must equal the development set; text hashes re-checked)
        -> rules_v1 spec hash          must equal the pinned value before anything is classified
        -> classify_scored_articles    unchanged KeywordEventClassifier (rules_v1), run twice
                                       in memory; both runs must agree
        -> write once, atomically      data/processed/events/phase11b/<scoring run id>.events-rules_v1.jsonl (+ .meta.json)
                                       data/results/research/phase11/11B/rules_v1_development_classification_<scoring run id>.json

Never called here: load_canonical_dataset, load_sentiment_dataset, load_event_dataset
(each reads every canonical record, sealed text included). Sealed records are never
turned into articles, never hashed as text, never classified and never reported
beyond their count and the Gate B id hash.

Event labels are keyword-rule output, not ground truth. No daily features are built here.

Usage (project root; operator-run only; no network, no model, no keys):

    python -m training.phase11b_event_classification
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.event_classifier import (  # noqa: E402
    KEYWORD_CLASSIFIER_NAME,
    KEYWORD_CLASSIFIER_VERSION,
    RULES,
    KeywordEventClassifier,
)
from services.event_taxonomy import EVENT_TYPES, IMPACT_FROM_SENTIMENT, NO_EVENT, TAXONOMY_VERSION  # noqa: E402
from services.historical_events import (  # noqa: E402
    EVENT_RECORD_VERSION,
    IMPACT_RULE,
    RECORD_ORDER,
    classifier_provenance,
    classify_scored_articles,
    data_quality_report,
    serialize_event_records,
)
from services.historical_sentiment import (  # noqa: E402
    PROVENANCE_FIELDS,
    SENTIMENT_DIR,
    HistoricalSentimentError,
    SentimentRecord,
    check_consistent_provenance,
)
from services.sentiment_features import ScoredArticle, SentimentFeatureError  # noqa: E402
from training.build_dataset import meta_path_for  # noqa: E402
from training.phase11b_sentiment_scoring import (  # noqa: E402
    APPROVED_MODEL,
    APPROVED_REVISION,
    BOUNDARY_RULE,
    CUTOFF_TEXT,
    DEVELOPMENT_CUTOFF_UTC,
    PHASE_11B_DIR,
    REPORT_PATH,
    SENTIMENT_DIR_11B,
    ScoringError,
    _atomic_write,
    _json_bytes,
    _sha256_file,
    load_integrity_report,
    partition_canonical,
    run_id_for,
    verify_frozen_artifacts,
    verify_partition_against_report,
)

STAGE = "11B.2"
CLASSIFICATION_VERSION = "phase11b_development_rules_v1_v1"
RULES_V1_SPEC_SHA256 = "974016388088b7eb50c8ed2659c738bc464d10a34c7b068471fd6a24d7cd44ad"
EVENTS_DIR_11B = PROJECT_ROOT / "data" / "processed" / "events" / "phase11b"
EVENTS_DIR_PHASE8 = PROJECT_ROOT / "data" / "processed" / "events"
SCORING_RECORD_TYPE = "phase_11b_finbert_development_scoring_run"


class ClassificationError(ScoringError):
    """Fail-closed signal for Stage 11B.2 event classification."""


# ============================================================
# rules_v1 identity
# ============================================================


def rules_v1_spec_sha256() -> str:
    """Content hash of the rules_v1 specification: classifier identity, taxonomy, every rule, impact mapping."""
    spec = {"classifier_name": KEYWORD_CLASSIFIER_NAME, "classifier_version": KEYWORD_CLASSIFIER_VERSION,
            "taxonomy_version": TAXONOMY_VERSION, "event_types": list(EVENT_TYPES), "no_event": NO_EVENT,
            "rules": {t: [list(r) for r in RULES[t]] for t in RULES}, "impact_from_sentiment": IMPACT_FROM_SENTIMENT}
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def verify_rules_v1() -> str:
    live = rules_v1_spec_sha256()
    if live != RULES_V1_SPEC_SHA256:
        raise ClassificationError(f"rules_v1 specification hash {live} differs from the pinned {RULES_V1_SPEC_SHA256}; "
                                  "refusing to classify")
    return live


# ============================================================
# Inputs: scoring run record and development score file
# ============================================================


def load_scoring_record(record_path: Path, *, report: dict, report_path: Path, frozen: dict, run_id: str) -> dict:
    if not record_path.is_file():
        raise ClassificationError(f"11B.2 scoring run record not found: {record_path.name}")
    rec = json.loads(record_path.read_text(encoding="utf-8"))
    model = rec.get("model") or {}
    checks = {
        "document_type": rec.get("document_type") == SCORING_RECORD_TYPE,
        "stage": rec.get("stage") == STAGE,
        "run_id": rec.get("run_id") == run_id,
        "frozen_artifact_sha256": rec.get("frozen_artifact_sha256") == frozen,
        "integrity_report.sha256": (rec.get("integrity_report") or {}).get("sha256") == _sha256_file(report_path),
        "canonical_input.sha256": (rec.get("canonical_input") or {}).get("sha256") == report["canonical_input"]["sha256"],
        "development_cutoff_utc": rec.get("development_cutoff_utc") == CUTOFF_TEXT,
        "boundary_rule": rec.get("boundary_rule") == BOUNDARY_RULE,
        "partition_reproduces_integrity_report": rec.get("partition_reproduces_integrity_report") is True,
        "development_article_count": rec.get("development_article_count") == report["development_article_count"],
        "sealed_article_count": rec.get("sealed_article_count") == report["sealed_article_count"],
        "scored_article_count": rec.get("scored_article_count") == report["development_article_count"],
        "excluded_article_count": rec.get("excluded_article_count") == 0,
        "phase_8_score_reuse": rec.get("phase_8_score_reuse") is False,
        "sealed_content_sent_to_scoring": rec.get("sealed_content_sent_to_scoring") is False,
        "model_name": model.get("model_name") == APPROVED_MODEL,
        "model_revision": model.get("model_revision") == APPROVED_REVISION,
        "outputs.scores.file": ((rec.get("outputs") or {}).get("scores") or {}).get("file") == f"{run_id}.sentiment.jsonl",
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise ClassificationError(f"11B.2 scoring run record fails verification: {failed}")
    return rec


def verify_score_file(sentiment_dir: Path, scoring_record: dict) -> tuple[Path, dict]:
    scores = scoring_record["outputs"]["scores"]
    path = sentiment_dir / scores["file"]
    if not path.is_file() or not meta_path_for(path).is_file():
        raise ClassificationError(f"development score file or its metadata not found: {path.name}")
    meta = json.loads(meta_path_for(path).read_text(encoding="utf-8"))
    if _sha256_file(path) != scores["sha256"] or meta.get("sha256") != scores["sha256"]:
        raise ClassificationError("development score file SHA256 does not match the 11B.2 scoring run record")
    if meta.get("rows") != scores["rows"] or \
            (meta.get("source_dataset") or {}).get("sha256") != scoring_record["canonical_input"]["sha256"]:
        raise ClassificationError("development score metadata does not match the 11B.2 scoring run record")
    expected = {k: scoring_record["model"][k] for k in PROVENANCE_FIELDS}
    if meta.get("provenance") != expected:
        raise ClassificationError("development score provenance does not match the 11B.2 scoring run record")
    return path, meta


def join_development_scores(score_path: Path, development: list, provenance: dict) -> list[ScoredArticle]:
    """Score records joined to development articles only; the two key sets must be identical."""
    by_key = {a.key: a for a in development}
    scored: dict[tuple[str, str], ScoredArticle] = {}
    with open(score_path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rec = SentimentRecord.from_dict(json.loads(line))
            except (HistoricalSentimentError, ValueError, KeyError, TypeError) as e:
                raise ClassificationError(f"score line {n} is invalid ({type(e).__name__})") from None
            if rec.key in scored:
                raise ClassificationError(f"duplicate score record {rec.key[0]}:{rec.key[1]}")
            article = by_key.get(rec.key)
            if article is None:
                raise ClassificationError(f"score record {rec.key[0]}:{rec.key[1]} is not a development article")
            if (article.symbols, article.information_available_at) != (rec.symbols, rec.information_available_at):
                raise ClassificationError(f"score record {rec.key[0]}:{rec.key[1]} does not match its article")
            try:
                check_consistent_provenance([rec.sentiment], provenance)
                scored[rec.key] = ScoredArticle(article, rec.sentiment)   # re-checks the text hash
            except (HistoricalSentimentError, SentimentFeatureError) as e:
                raise ClassificationError(f"score record {rec.key[0]}:{rec.key[1]} rejected: {e}") from None
    if set(scored) != set(by_key):
        raise ClassificationError(f"score article set differs from the development set "
                                  f"({len(set(by_key) - set(scored))} missing, {len(set(scored) - set(by_key))} extra)")
    return list(scored.values())


def classify_development(scored: list[ScoredArticle], development_keys: set[str]):
    """Re-check development membership, then classify twice with fresh classifiers; results must agree."""
    for s in scored:
        a = s.article
        if a.information_available_at > DEVELOPMENT_CUTOFF_UTC or a.article_key not in development_keys:
            raise ClassificationError(f"{a.article_key} is not a development article; refusing to classify")
    first_classifier = KeywordEventClassifier()
    first = classify_scored_articles(scored, first_classifier)
    second = classify_scored_articles(scored, KeywordEventClassifier())
    if [(c.key, c.prediction) for c in first] != [(c.key, c.prediction) for c in second]:
        raise ClassificationError("rules_v1 classification is not deterministic across two runs")
    return first, first_classifier


# ============================================================
# End to end
# ============================================================


def run_development_classification(*, report_path: Path = REPORT_PATH, sentiment_dir: Path = SENTIMENT_DIR_11B,
                                   output_dir: Path = EVENTS_DIR_11B, record_dir: Path = PHASE_11B_DIR,
                                   clock: Callable[[], datetime] | None = None) -> dict:
    clock = clock or (lambda: datetime.now(timezone.utc))
    report_path, sentiment_dir = Path(report_path), Path(sentiment_dir)
    output_dir, record_dir = Path(output_dir), Path(record_dir)
    if sentiment_dir.resolve() == Path(SENTIMENT_DIR).resolve():
        raise ClassificationError("sentiment_dir is the Phase 8 sentiment directory; only Phase 11B scores are accepted")
    if output_dir.resolve() == EVENTS_DIR_PHASE8.resolve():
        raise ClassificationError("output_dir is the Phase 8 events directory; Phase 11B events go to a separate directory")

    frozen = verify_frozen_artifacts()
    rules_sha = verify_rules_v1()
    report = load_integrity_report(report_path)
    canonical_sha = report["canonical_input"]["sha256"]
    run_id = run_id_for(canonical_sha)
    record_path = record_dir / f"rules_v1_development_classification_{run_id}.json"
    if record_path.exists():
        raise ClassificationError(f"run record already exists ({record_path.name}); this run is complete and write-once")

    scoring_record_path = record_dir / f"finbert_development_scoring_{run_id}.json"
    scoring_record = load_scoring_record(scoring_record_path, report=report, report_path=report_path,
                                         frozen=frozen, run_id=run_id)
    score_path, score_meta = verify_score_file(sentiment_dir, scoring_record)

    canonical_path = Path(report["canonical_input"]["path"])
    if not canonical_path.is_absolute():
        canonical_path = PROJECT_ROOT / canonical_path
    if not canonical_path.is_file():
        raise ClassificationError(f"canonical dataset named by the integrity report not found: {canonical_path.name}")

    started = clock()
    partition = partition_canonical(canonical_path, expected_sha256=canonical_sha)
    verify_partition_against_report(partition, report)
    development, development_keys = partition.development, partition.development_keys
    sealed_count = len(partition.sealed_keys)
    del partition
    if scoring_record["development_article_sha256"] != report["development_article_sha256"]:
        raise ClassificationError("11B.2 scoring run record development hash differs from the integrity report")

    provenance = {k: scoring_record["model"][k] for k in PROVENANCE_FIELDS}
    scored = join_development_scores(score_path, development, provenance)
    del development
    classified, classifier = classify_development(scored, development_keys)

    payload = serialize_event_records(classified, classifier_provenance(classifier))
    events_path = output_dir / f"{run_id}.events-{KEYWORD_CLASSIFIER_VERSION}.jsonl"
    if events_path.exists() and events_path.read_bytes() != payload:
        raise ClassificationError(f"{events_path.name} exists with different content; refusing to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(events_path, payload)
    events_sha = hashlib.sha256(payload).hexdigest()
    finished = clock()
    quality = data_quality_report(classified, len(scored))
    meta = {
        "record_version": EVENT_RECORD_VERSION,
        "classifier": classifier_provenance(classifier),
        "impact_source": {"file": score_path.name, "sha256": scoring_record["outputs"]["scores"]["sha256"],
                          "rule": IMPACT_RULE, "sentiment_provenance": score_meta.get("provenance")},
        "source_dataset": score_meta.get("source_dataset"),
        "labels_are_ground_truth": False,
        "data_quality": quality,
        "rows": len(classified),
        "ordering": RECORD_ORDER,
        "sha256": events_sha,
        "generated_at": finished.isoformat(timespec="seconds"),
        "phase11b": {"stage": STAGE, "scoring_run_id": run_id, "classification_version": CLASSIFICATION_VERSION,
                     "rules_v1_spec_sha256": rules_sha, "partition": "development only",
                     "development_cutoff_utc": CUTOFF_TEXT},
    }
    _atomic_write(meta_path_for(events_path), _json_bytes(meta))

    record = {
        "document_type": "phase_11b_rules_v1_development_classification_run",
        "phase": "11B", "stage": STAGE, "classification_version": CLASSIFICATION_VERSION, "run_id": run_id,
        "started_at": started.isoformat(), "finished_at": finished.isoformat(),
        "frozen_artifact_sha256": frozen,
        "rules_v1_spec_sha256": rules_sha,
        "classifier": classifier_provenance(classifier),
        "integrity_report": {"file": report_path.name, "sha256": _sha256_file(report_path)},
        "canonical_input": {"file": canonical_path.name, "sha256": canonical_sha,
                            "rows": report["canonical_input"].get("rows")},
        "scoring_run_record": {"file": scoring_record_path.name, "sha256": _sha256_file(scoring_record_path)},
        "score_input": {"file": score_path.name, "sha256": scoring_record["outputs"]["scores"]["sha256"],
                        "rows": scoring_record["outputs"]["scores"]["rows"], "provenance": provenance},
        "development_cutoff_utc": CUTOFF_TEXT, "boundary_rule": BOUNDARY_RULE,
        "partition_reproduces_integrity_report": True,
        "development_article_count": len(development_keys), "sealed_article_count": sealed_count,
        "development_article_sha256": report["development_article_sha256"],
        "classified_article_count": len(classified),
        "excluded_article_count": 0,
        "exclusion_policy": "none; an article that cannot be joined, validated or classified fails the whole run",
        "determinism_check": "two independent in-memory classifications identical",
        "n_events": quality["n_events"], "n_other": quality["n_other"],
        "labels_are_ground_truth": False,
        "phase_8_score_reuse": False,
        "sealed_content_sent_to_classifier": False,
        "outputs": {"events": {"file": events_path.name, "sha256": events_sha, "rows": len(classified)},
                    "events_meta": meta_path_for(events_path).name},
    }
    if record_path.exists():
        raise ClassificationError(f"run record already exists ({record_path.name}); refusing to overwrite")
    record_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(record_path, _json_bytes(record))
    return record


# ============================================================
# CLI (operator-run only)
# ============================================================


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(
        description="Phase 11B Stage 11B.2: rules_v1 event classification of the DEVELOPMENT partition only. "
                    "All inputs come from the Gate B integrity report and the 11B.2 scoring run record; "
                    "no paths, dates or article selections can be passed.").parse_args(argv)
    record = run_development_classification()
    print(f"Run id      : {record['run_id']}")
    print(f"Development : {record['development_article_count']} classified "
          f"({record['n_events']} with event evidence, {record['n_other']} OTHER)")
    print(f"Sealed      : {record['sealed_article_count']} (not classified)")
    print(f"rules_v1    : {record['rules_v1_spec_sha256'][:16]}...")
    print(f"Events      : {record['outputs']['events']['file']}  sha256 {record['outputs']['events']['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
