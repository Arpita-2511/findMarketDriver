"""
Phase 11B Stage 11B.2: development-only FinBERT scoring.

    canonical multi-symbol dataset (Stage 11B.1, sha256 recorded by Gate B)
        -> partition_canonical        stream; per record read ONLY provider,
                                      provider_article_id, created_at, updated_at,
                                      information_available_at; frozen split
                                      information_available_at <= 2024-09-30T20:30:00Z;
                                      sealed records keep only their key (count/hash)
        -> verify vs integrity report counts and both partition hashes must match
        -> score_development_subset   re-checks every article is development, then
                                      the unchanged Phase 5A/8 scoring
                                      (services.historical_sentiment.score_canonical_articles)
        -> write once, atomically     data/processed/sentiment/phase11b/<run_id>.sentiment.jsonl (+ .meta.json)
                                      data/results/research/phase11/11B/finbert_development_scoring_<run_id>.json

Sealed records are never turned into NewsArticle objects, never reach text
construction, FinBERT, classification, features or any report, and are never
logged; only their article_key is kept, to reproduce the Gate B count and hash.

Model: ProsusAI/finbert at the frozen revision 4556d13015211d73dccd3fdd39d39232506f3e43
only. Phase 8 scores are never reused: the checkpoint lives under the Phase 11B
directory, is bound to this run's identity (canonical sha256, development hash,
cutoff, frozen policy hashes, model provenance) and is refused on any mismatch.

Usage (project root; operator-run only; model must already be cached):

    set HF_HUB_OFFLINE=1
    python -m training.phase11b_sentiment_scoring --canonical data/processed/news/<run_id>.jsonl
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
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.alpaca_news_provider import (  # noqa: E402
    MALFORMED_POLICY_AMENDMENT_SHA256,
    RUN_CAP_POLICY_AMENDMENT_SHA256,
)
from services.finbert_sentiment import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    FINBERT_MODEL,
    LABELS,
    FinBertSentimentService,
)
from services.historical_sentiment import (  # noqa: E402
    PINNED_FINBERT_REVISION,
    RECORD_ORDER,
    SENTIMENT_DIR,
    SENTIMENT_RECORD_VERSION,
    backend_provenance,
    check_consistent_provenance,
    load_checkpoint,
    score_canonical_articles,
    serialize_records,
)
from services.news_schema import (  # noqa: E402
    NewsArticle,
    NewsValidationError,
    historical_information_available_at,
    parse_timestamp,
)
from training.build_dataset import meta_path_for  # noqa: E402
from training.phase11b_ingest_closeout import (  # noqa: E402
    AMENDMENT_04_PATH,
    AMENDMENT_05_PATH,
    DESIGN_FREEZE_PATH,
    DEVELOPMENT_CUTOFF_UTC,
    FROZEN_DESIGN_FREEZE_SHA256,
    PHASE_11B_DIR,
    REPORT_PATH,
    _development_article_sha256,
    _sealed_article_id_sha256,
)

STAGE = "11B.2"
SCORING_VERSION = "phase11b_development_finbert_v1"
APPROVED_MODEL = "ProsusAI/finbert"
APPROVED_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
CUTOFF_TEXT = "2024-09-30T20:30:00Z"
BOUNDARY_RULE = "information_available_at <= cutoff -> development; > cutoff -> sealed"
SENTIMENT_DIR_11B = SENTIMENT_DIR / "phase11b"
AMENDMENT_03_PATH = PHASE_11B_DIR / "amendment_03_test_gates.json"
FROZEN_ARTIFACTS = {
    "design_freeze": (DESIGN_FREEZE_PATH, FROZEN_DESIGN_FREEZE_SHA256),
    "amendment_03": (AMENDMENT_03_PATH, "0040483259cd3febea1feff6301fa545acaa8a21813006f62a06a50d49e02253"),
    "amendment_04": (AMENDMENT_04_PATH, MALFORMED_POLICY_AMENDMENT_SHA256),
    "amendment_05": (AMENDMENT_05_PATH, RUN_CAP_POLICY_AMENDMENT_SHA256),
}
PARTITION_FIELDS = frozenset({"provider", "provider_article_id", "created_at", "updated_at",
                              "information_available_at"})

assert APPROVED_MODEL == FINBERT_MODEL and APPROVED_REVISION == PINNED_FINBERT_REVISION
assert DEVELOPMENT_CUTOFF_UTC == datetime(2024, 9, 30, 20, 30, tzinfo=timezone.utc)


class ScoringError(ValueError):
    """Fail-closed signal for Stage 11B.2 scoring."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ============================================================
# Frozen inputs
# ============================================================


def verify_frozen_artifacts() -> dict[str, str]:
    """Design freeze and Amendments 03-05 must match their pins; the freeze must name this model and cutoff."""
    hashes = {}
    for name, (path, pinned) in FROZEN_ARTIFACTS.items():
        if not Path(path).is_file() or _sha256_file(path) != pinned:
            raise ScoringError(f"{name} ({Path(path).name}) is missing or differs from its pinned SHA256")
        hashes[name] = pinned
    design = json.loads(Path(DESIGN_FREEZE_PATH).read_text(encoding="utf-8"))
    fb, sealing = design["finbert_provenance"], design["sealing_policy"]
    if (fb["model"], fb["revision"], fb["phase_8_score_reuse"]) != (APPROVED_MODEL, APPROVED_REVISION, False):
        raise ScoringError("design freeze FinBERT provenance does not match the approved model, revision or reuse policy")
    if sealing["development_cutoff_utc"] != CUTOFF_TEXT or \
            sealing["development_set"]["definition"] != "information_available_at <= development_cutoff_utc":
        raise ScoringError("design freeze development cutoff or partition rule differs from this module")
    return hashes


def load_integrity_report(report_path: Path) -> dict:
    """The Gate B integrity report (aggregate metadata only) and the facts this stage depends on."""
    report_path = Path(report_path)
    if not report_path.is_file():
        raise ScoringError(f"integrity report not found: {report_path}")
    r = json.loads(report_path.read_text(encoding="utf-8"))
    checks = {
        "split_applied_before_scoring": r.get("split_applied_before_scoring") is True,
        "development_sealed_intersection_empty": r.get("development_sealed_intersection_empty") is True,
        "development_cutoff_utc": r.get("development_cutoff_utc") == CUTOFF_TEXT,
        "boundary_rule": r.get("boundary_rule") == BOUNDARY_RULE,
        "malformed_policy_amendment_sha256": r.get("malformed_policy_amendment_sha256") == MALFORMED_POLICY_AMENDMENT_SHA256,
        "run_cap_policy_amendment_sha256": r.get("run_cap_policy_amendment_sha256") == RUN_CAP_POLICY_AMENDMENT_SHA256,
        "run_cap_evaluation": r.get("run_cap_evaluation") == "final_totals",
        "development_article_count": isinstance(r.get("development_article_count"), int),
        "sealed_article_count": isinstance(r.get("sealed_article_count"), int),
        "canonical_input.sha256": isinstance((r.get("canonical_input") or {}).get("sha256"), str),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise ScoringError(f"integrity report fails Stage 11B.2 prerequisites: {failed}")
    return r


# ============================================================
# Partition (streaming; sealed records keep only their key)
# ============================================================


def _keep_partition_fields(pairs):
    return {k: v for k, v in pairs if k in PARTITION_FIELDS}


@dataclass(frozen=True)
class Partition:
    development: list[NewsArticle]
    sealed_keys: list[str]
    file_sha256: str

    @property
    def development_keys(self) -> set[str]:
        return {a.article_key for a in self.development}


def partition_canonical(canonical_path: Path, *, expected_sha256: str) -> Partition:
    """
    Stream the canonical dataset once. For each record only the partition
    fields are retained; the frozen rule decides the side. Development records
    are then parsed in full; sealed records keep nothing but provider:id.
    The file hash is verified before anything is returned.
    """
    h = hashlib.sha256()
    development: list[NewsArticle] = []
    sealed_keys: list[str] = []
    seen: set[str] = set()
    with open(canonical_path, "rb") as f:
        for n, raw in enumerate(f, start=1):
            h.update(raw)
            if not raw.strip():
                continue
            try:
                view = json.loads(raw, object_pairs_hook=_keep_partition_fields)
            except ValueError:
                raise ScoringError(f"canonical line {n} is not valid JSON") from None
            provider, article_id = view.get("provider"), view.get("provider_article_id")
            if not (isinstance(provider, str) and provider.strip() and isinstance(article_id, str) and article_id.strip()):
                raise ScoringError(f"canonical line {n} has an invalid article identifier")
            key = f"{provider}:{article_id}"
            if key in seen:
                raise ScoringError(f"duplicate article identifier {key} in canonical dataset")
            seen.add(key)
            try:
                created = parse_timestamp(view.get("created_at"), "created_at")
                updated = parse_timestamp(view["updated_at"], "updated_at") if view.get("updated_at") else None
                recorded = parse_timestamp(view.get("information_available_at"), "information_available_at")
            except NewsValidationError as e:
                raise ScoringError(f"{key}: {e}") from None
            available = historical_information_available_at(created, updated)
            if available != recorded:
                raise ScoringError(f"{key}: information_available_at is not max(created_at, updated_at)")
            if available <= DEVELOPMENT_CUTOFF_UTC:
                try:
                    article = NewsArticle.from_dict(json.loads(raw))
                except (NewsValidationError, ValueError, KeyError, TypeError) as e:
                    raise ScoringError(f"{key}: invalid development record ({type(e).__name__})") from None
                if article.article_key != key or article.information_available_at != available:
                    raise ScoringError(f"{key}: development record does not match its partition fields")
                development.append(article)
            else:
                sealed_keys.append(key)
    digest = h.hexdigest()
    if digest != expected_sha256:
        raise ScoringError("canonical dataset SHA256 does not match the integrity report; refusing to score")
    return Partition(development, sealed_keys, digest)


def verify_partition_against_report(partition: Partition, report: dict) -> None:
    """The streamed split must reproduce Gate B's counts and both partition hashes exactly."""
    if partition.development_keys & set(partition.sealed_keys):
        raise ScoringError("development and sealed identifiers intersect")
    mismatches = []
    if len(partition.development) != report["development_article_count"]:
        mismatches.append("development_article_count")
    if len(partition.sealed_keys) != report["sealed_article_count"]:
        mismatches.append("sealed_article_count")
    if _sealed_article_id_sha256(partition.sealed_keys) != report.get("sealed_article_id_sha256"):
        mismatches.append("sealed_article_id_sha256")
    if _development_article_sha256(partition.development) != report.get("development_article_sha256"):
        mismatches.append("development_article_sha256")
    if mismatches:
        raise ScoringError(f"partition does not reproduce the integrity report: {mismatches}")


# ============================================================
# Model, checkpoint, scoring
# ============================================================


def check_model(service: FinBertSentimentService) -> dict:
    backend = service.backend
    if backend.model_name != APPROVED_MODEL:
        raise ScoringError(f"model {backend.model_name!r} is not the approved {APPROVED_MODEL!r}")
    if backend.model_revision != APPROVED_REVISION:
        raise ScoringError(f"model revision {backend.model_revision!r} is not the pinned {APPROVED_REVISION}")
    if sorted(str(lbl).lower() for lbl in backend.labels) != sorted(LABELS):
        raise ScoringError(f"model classes {tuple(backend.labels)!r} are not {LABELS!r}")
    return backend_provenance(service)


def run_id_for(canonical_sha256: str) -> str:
    return f"phase11b_dev_{canonical_sha256[:12]}_cut20240930T203000Z_finbert-{APPROVED_REVISION[:12]}"


def _atomic_write(path: Path, payload: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _json_bytes(obj: dict) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def prepare_checkpoint(output_dir: Path, run_id: str, binding: dict) -> Path:
    """
    Checkpoint for this run only. A checkpoint without its binding, or with a
    binding for any other input/model/policy, is refused - never reused.
    """
    checkpoint = output_dir / f"{run_id}.checkpoint.jsonl"
    binding_path = output_dir / f"{run_id}.checkpoint.binding.json"
    if binding_path.exists():
        if json.loads(binding_path.read_text(encoding="utf-8")) != binding:
            raise ScoringError("checkpoint binding does not match this run; refusing to reuse it")
    elif checkpoint.exists():
        raise ScoringError("checkpoint exists without a binding; refusing to reuse it")
    else:
        _atomic_write(binding_path, _json_bytes(binding))
    return checkpoint


def score_development_subset(articles: list[NewsArticle], development_keys: set[str],
                             service: FinBertSentimentService, *, checkpoint: Path | None):
    """Every article must be development (frozen rule and partition membership) before any text is built."""
    for a in articles:
        if a.information_available_at > DEVELOPMENT_CUTOFF_UTC or a.article_key not in development_keys:
            raise ScoringError(f"{a.article_key} is not a development article; refusing to score")
    if checkpoint is not None and checkpoint.exists():
        cached = load_checkpoint(checkpoint, backend_provenance(service))
        foreign = {f"{key[0]}:{key[1]}" for key, _text_hash in cached} - development_keys
        if foreign:
            raise ScoringError(f"checkpoint contains {len(foreign)} non-development record(s); refusing to reuse it")
    return score_canonical_articles(articles, service, checkpoint=checkpoint)


# ============================================================
# End to end
# ============================================================


def run_development_scoring(canonical_path: Path, service_factory: Callable[[], FinBertSentimentService], *,
                            report_path: Path = REPORT_PATH, output_dir: Path = SENTIMENT_DIR_11B,
                            record_dir: Path = PHASE_11B_DIR, operator: dict | None = None,
                            clock: Callable[[], datetime] | None = None) -> dict:
    clock = clock or (lambda: datetime.now(timezone.utc))
    canonical_path, output_dir, record_dir = Path(canonical_path), Path(output_dir), Path(record_dir)
    if output_dir.resolve() == Path(SENTIMENT_DIR).resolve():
        raise ScoringError("output_dir is the Phase 8 sentiment directory; Phase 11B scores go to a separate directory")

    frozen = verify_frozen_artifacts()
    report = load_integrity_report(report_path)
    expected_sha = report["canonical_input"]["sha256"]
    meta_file = meta_path_for(canonical_path)
    if not meta_file.is_file() or json.loads(meta_file.read_text(encoding="utf-8")).get("sha256") != expected_sha:
        raise ScoringError("canonical dataset metadata is missing or does not match the integrity report")

    run_id = run_id_for(expected_sha)
    record_path = record_dir / f"finbert_development_scoring_{run_id}.json"
    if record_path.exists():
        raise ScoringError(f"run record already exists ({record_path.name}); this run is complete and write-once")

    started = clock()
    partition = partition_canonical(canonical_path, expected_sha256=expected_sha)
    verify_partition_against_report(partition, report)
    development, dev_keys = partition.development, partition.development_keys
    sealed_count = len(partition.sealed_keys)
    del partition                                                       # sealed keys are not needed past here
    if not development:
        raise ScoringError("no development articles to score")
    dev_sha = _development_article_sha256(development)

    service = service_factory()                                         # model loaded only after the split is verified
    provenance = check_model(service)
    output_dir.mkdir(parents=True, exist_ok=True)
    binding = {"scoring_version": SCORING_VERSION, "run_id": run_id, "canonical_sha256": expected_sha,
               "development_article_sha256": dev_sha, "development_cutoff_utc": CUTOFF_TEXT,
               "frozen_artifact_sha256": frozen, "provenance": provenance}
    checkpoint = prepare_checkpoint(output_dir, run_id, binding)

    scored, stats = score_development_subset(development, dev_keys, service, checkpoint=checkpoint)
    if {s.article.article_key for s in scored} != dev_keys or len(scored) != len(dev_keys):
        raise ScoringError("scored set does not equal the development set")
    check_consistent_provenance((s.sentiment for s in scored), provenance)

    payload = serialize_records(scored)                                 # sentiment_records_v1; no article text
    scores_path = output_dir / f"{run_id}.sentiment.jsonl"
    if scores_path.exists() and scores_path.read_bytes() != payload:
        raise ScoringError(f"{scores_path.name} exists with different content; refusing to overwrite")
    _atomic_write(scores_path, payload)
    scores_sha = hashlib.sha256(payload).hexdigest()
    finished = clock()
    meta = {
        "record_version": SENTIMENT_RECORD_VERSION,
        "source_dataset": {"file": canonical_path.name, "sha256": expected_sha,
                           "run_id": report["canonical_input"].get("run_id")},
        "provenance": provenance,
        **stats,
        "rows": len(scored),
        "ordering": RECORD_ORDER,
        "sha256": scores_sha,
        "generated_at": finished.isoformat(timespec="seconds"),
        "phase11b": {"stage": STAGE, "run_id": run_id, "scoring_version": SCORING_VERSION,
                     "partition": "development only", "development_cutoff_utc": CUTOFF_TEXT},
    }
    _atomic_write(meta_path_for(scores_path), _json_bytes(meta))

    record = {
        "document_type": "phase_11b_finbert_development_scoring_run",
        "phase": "11B", "stage": STAGE, "scoring_version": SCORING_VERSION, "run_id": run_id,
        "started_at": started.isoformat(), "finished_at": finished.isoformat(),
        "frozen_artifact_sha256": frozen,
        "integrity_report": {"file": Path(report_path).name, "sha256": _sha256_file(report_path)},
        "canonical_input": {"file": canonical_path.name, "sha256": expected_sha,
                            "rows": report["canonical_input"].get("rows")},
        "development_cutoff_utc": CUTOFF_TEXT, "boundary_rule": BOUNDARY_RULE,
        "partition_reproduces_integrity_report": True,
        "development_article_count": len(development), "sealed_article_count": sealed_count,
        "development_article_sha256": dev_sha,
        "eligible_article_count": len(development),
        "scored_article_count": len(scored),
        "scored_from_checkpoint_count": stats["n_from_checkpoint"],
        "excluded_article_count": 0,
        "exclusion_policy": "none; an article that cannot be scored or validated fails the whole run",
        "model": {**provenance, "approved_model": APPROVED_MODEL, "approved_revision": APPROVED_REVISION,
                  "labels": list(LABELS), "batch_size": service.batch_size},
        "phase_8_score_reuse": False,
        "sealed_content_sent_to_scoring": False,
        "outputs": {"scores": {"file": scores_path.name, "sha256": scores_sha, "rows": len(scored)},
                    "scores_meta": meta_path_for(scores_path).name,
                    "checkpoint": checkpoint.name},
        "operator": operator or {},
    }
    if record_path.exists():
        raise ScoringError(f"run record already exists ({record_path.name}); refusing to overwrite")
    record_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(record_path, _json_bytes(record))
    return record


# ============================================================
# CLI (operator-run only)
# ============================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 11B Stage 11B.2: FinBERT scoring of the DEVELOPMENT partition only. "
                    "No date range, cutoff or article selection can be passed: the frozen partition decides.")
    parser.add_argument("--canonical", type=Path, required=True, help="Phase 11B canonical news dataset (.jsonl)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args(argv)

    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit("HF_HUB_OFFLINE=1 is required: the pinned FinBERT revision must load from the local cache "
                         "and no download may happen during scoring")
    record = run_development_scoring(
        args.canonical,
        lambda: FinBertSentimentService.load(revision=APPROVED_REVISION, device=args.device,
                                             batch_size=args.batch_size),
        operator={"hf_hub_offline": True, "device": args.device})
    print(f"Run id      : {record['run_id']}")
    print(f"Development : {record['development_article_count']} eligible, {record['scored_article_count']} scored "
          f"({record['scored_from_checkpoint_count']} from checkpoint)")
    print(f"Sealed      : {record['sealed_article_count']} (not scored)")
    print(f"Scores      : {record['outputs']['scores']['file']}  sha256 {record['outputs']['scores']['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
