"""
Phase 11B Stage 11B.2: development-only rules_v1 event classification - synthetic tests.

Every input is synthetic: a synthetic canonical dataset and integrity report, scored
by the real 11B.2 scoring pipeline with a fake FinBERT backend. No real article,
no real score file and no real canonical dataset is touched.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from datetime import timedelta
from pathlib import Path

import pytest

import services.event_classifier as event_classifier
import services.historical_events as historical_events
import services.historical_sentiment as historical_sentiment
import services.news_dataset as news_dataset
import training.phase11b_event_classification as ec
from services.event_classifier import KeywordEventClassifier
from services.historical_events import load_event_dataset
from services.news_schema import NewsArticle
from tests.sentiment_fakes import news
from tests.unit.test_phase11b_sentiment_scoring import CUT, H, US, fake, make_env, run as run_scoring

SENTINEL = "SEALED_EVENT_SENTINEL_text_must_never_reach_the_classifier"


# ============================================================
# Builders
# ============================================================


def dev_articles():
    """Synthetic development headlines chosen to exercise several rules_v1 types."""
    return [
        news(1, CUT - 48 * H, headline="Acme beats EPS estimates in Q3 results"),
        news(2, CUT - 24 * H, headline="Analyst upgrades Acme, raises price target"),
        news(3, CUT - 2 * H, headline="Acme announces new board members at annual meeting"),
        news(4, CUT, headline="Acme unveils plant expansion with new supplier"),    # exactly at the cutoff
        news(5, CUT - H, headline="A quiet day with nothing specific"),
    ]


def scored_env(tmp_path, dev=None, sealed=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    env = make_env(tmp_path, dev=dev if dev is not None else dev_articles(),
                   sealed=sealed if sealed is not None else [news(9, CUT + H, headline="Acme beats EPS estimates")])
    run_scoring(env, fake())                                                       # synthetic 11B.2 outputs
    env.events = tmp_path / "events" / "phase11b"
    return env


def classify(env, **kw):
    return ec.run_development_classification(report_path=env.report, sentiment_dir=env.out,
                                             output_dir=env.events, record_dir=env.records, **kw)


def events_path(env):
    return env.events / f"{env.run_id}.events-rules_v1.jsonl"


def event_ids(env):
    return [json.loads(line)["provider_article_id"]
            for line in events_path(env).read_text(encoding="utf-8").splitlines()]


def no_event_outputs(env):
    events = list(env.events.glob("*.jsonl")) if env.events.exists() else []
    records = list(env.records.glob("rules_v1_development_classification_*.json"))
    return not events and not records


def scoring_record_path(env):
    return env.records / f"finbert_development_scoring_{env.run_id}.json"


def score_path(env):
    return env.out / f"{env.run_id}.sentiment.jsonl"


def reseal_scores(env, mutate_lines):
    """Rewrite the score file and update its metadata and the scoring run record so all hashes agree."""
    lines = score_path(env).read_text(encoding="utf-8").splitlines()
    payload = "".join(line + "\n" for line in mutate_lines(lines)).encode("utf-8")
    score_path(env).write_bytes(payload)
    sha, rows = hashlib.sha256(payload).hexdigest(), payload.count(b"\n")
    meta_file = score_path(env).with_suffix(".meta.json")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta.update(sha256=sha, rows=rows)
    meta_file.write_text(json.dumps(meta), encoding="utf-8")
    rec = json.loads(scoring_record_path(env).read_text(encoding="utf-8"))
    rec["outputs"]["scores"].update(sha256=sha, rows=rows)
    scoring_record_path(env).write_text(json.dumps(rec), encoding="utf-8")


def edit_scoring_record(env, **changes):
    rec = json.loads(scoring_record_path(env).read_text(encoding="utf-8"))
    rec.update(changes)
    scoring_record_path(env).write_text(json.dumps(rec), encoding="utf-8")


@pytest.fixture
def banned_loaders(monkeypatch):
    """Any call to a loader that reads the whole canonical dataset fails the test."""
    def boom(*_a, **_k):
        raise AssertionError("a full-canonical loader was called by the development-only command")
    for module, name in [(news_dataset, "load_canonical_dataset"), (historical_sentiment, "load_canonical_dataset"),
                         (historical_sentiment, "load_sentiment_dataset"), (historical_events, "load_canonical_dataset"),
                         (historical_events, "load_sentiment_dataset"), (historical_events, "load_event_dataset")]:
        monkeypatch.setattr(module, name, boom)


# ============================================================
# Happy path, cutoff boundary, rules hash, run record
# ============================================================


def test_development_articles_are_classified_with_rules_v1(tmp_path, banned_loaders):
    env = scored_env(tmp_path)
    record = classify(env)
    rows = {json.loads(line)["provider_article_id"]: json.loads(line)
            for line in events_path(env).read_text(encoding="utf-8").splitlines()}
    assert sorted(rows) == ["1", "2", "3", "4", "5"]
    assert rows["1"]["event_type"] == "EARNINGS" and rows["2"]["event_type"] == "ANALYST_RATING"
    assert rows["3"]["event_type"] == "MANAGEMENT_GOVERNANCE" and rows["4"]["event_type"] == "OPERATIONS"
    assert rows["5"]["event_type"] == "OTHER" and rows["5"]["event_confidence"] == 0.0
    assert all(r["classifier_version"] == "rules_v1" and r["taxonomy_version"] == "taxonomy_v1" for r in rows.values())
    assert record["rules_v1_spec_sha256"] == ec.RULES_V1_SPEC_SHA256
    assert (record["development_article_count"], record["sealed_article_count"],
            record["classified_article_count"], record["excluded_article_count"]) == (5, 1, 5, 0)
    assert record["outputs"]["events"]["sha256"] == hashlib.sha256(events_path(env).read_bytes()).hexdigest()
    assert record["score_input"]["sha256"] == hashlib.sha256(score_path(env).read_bytes()).hexdigest()
    assert record["phase_8_score_reuse"] is False and record["sealed_content_sent_to_classifier"] is False
    assert record["labels_are_ground_truth"] is False


def test_cutoff_equality_is_classified_and_one_microsecond_later_is_not(tmp_path, banned_loaders):
    env = scored_env(tmp_path, dev=[news(1, CUT, headline="Acme beats EPS estimates")],
                     sealed=[news(2, CUT + US, headline="Acme beats EPS estimates")])
    classify(env)
    assert event_ids(env) == ["1"]


def test_rules_v1_change_is_refused_before_classifying(tmp_path, monkeypatch):
    env = scored_env(tmp_path)
    monkeypatch.setitem(event_classifier.RULES, "MACRO", event_classifier.RULES["MACRO"][:1])
    with pytest.raises(ec.ClassificationError, match="rules_v1 specification hash"):
        classify(env)
    assert no_event_outputs(env)


def test_pinned_rules_hash_matches_the_current_specification():
    assert ec.rules_v1_spec_sha256() == ec.RULES_V1_SPEC_SHA256 == \
        "974016388088b7eb50c8ed2659c738bc464d10a34c7b068471fd6a24d7cd44ad"


# ============================================================
# Sealed records and banned loaders
# ============================================================


def test_sealed_text_never_reaches_classifier_outputs_exceptions_or_logs(tmp_path, monkeypatch, capsys):
    sealed = news(9, CUT + H).to_dict()
    sealed.update(headline=123, summary=SENTINEL, content=SENTINEL)              # a full parse would raise
    env = make_env(tmp_path, dev=dev_articles(), sealed=[sealed])
    run_scoring(env, fake())
    env.events = tmp_path / "events" / "phase11b"

    parsed, seen_texts = [], []
    original_from_dict, original_classify = NewsArticle.from_dict, KeywordEventClassifier.classify_texts
    monkeypatch.setattr(NewsArticle, "from_dict",
                        classmethod(lambda cls, d: (parsed.append(d.get("provider_article_id")), original_from_dict(d))[1]))
    monkeypatch.setattr(KeywordEventClassifier, "classify_texts",
                        lambda self, texts: (seen_texts.extend(texts), original_classify(self, texts))[1])
    record = classify(env)
    assert "9" not in parsed and sorted(parsed) == ["1", "2", "3", "4", "5"]
    assert len(seen_texts) == 10 and all(SENTINEL not in t for t in seen_texts)    # 5 articles, classified twice
    assert record["sealed_article_count"] == 1
    for path in [*env.events.iterdir(), *env.records.iterdir()]:
        assert SENTINEL.encode() not in path.read_bytes(), path.name
    assert SENTINEL not in capsys.readouterr().out


def test_failure_on_a_sealed_record_does_not_expose_its_text(tmp_path):
    env = scored_env(tmp_path)
    bad_sealed = news(10, CUT + H).to_dict()
    bad_sealed.update(created_at="not-a-date", summary=SENTINEL)
    with open(env.canonical, "a", encoding="utf-8") as f:                          # tampered canonical
        f.write(json.dumps(bad_sealed) + "\n")
    with pytest.raises(ec.ScoringError) as exc:
        classify(env)
    assert SENTINEL not in str(exc.value) and no_event_outputs(env)


def test_banned_loader_fixture_really_fires(banned_loaders):
    for call in (lambda: news_dataset.load_canonical_dataset("x"),
                 lambda: historical_sentiment.load_sentiment_dataset("x", "y"),
                 lambda: historical_events.load_event_dataset("x", "y", "z")):
        with pytest.raises(AssertionError, match="full-canonical loader"):
            call()


def test_module_never_imports_or_calls_full_canonical_loaders():
    tree = ast.parse(inspect.getsource(ec))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {"load_canonical_dataset", "load_sentiment_dataset", "load_event_dataset"}


# ============================================================
# Wrong or inconsistent inputs
# ============================================================


def test_missing_scoring_run_record_is_refused(tmp_path):
    env = scored_env(tmp_path)
    scoring_record_path(env).unlink()
    with pytest.raises(ec.ClassificationError, match="scoring run record not found"):
        classify(env)


@pytest.mark.parametrize("changes", [
    {"phase_8_score_reuse": True}, {"sealed_content_sent_to_scoring": True}, {"excluded_article_count": 1},
    {"development_cutoff_utc": "2024-09-30T20:30:01Z"}, {"stage": "8"}, {"partition_reproduces_integrity_report": False},
])
def test_scoring_run_record_that_fails_verification_is_refused(tmp_path, changes):
    env = scored_env(tmp_path)
    edit_scoring_record(env, **changes)
    with pytest.raises(ec.ClassificationError, match="scoring run record fails verification"):
        classify(env)
    assert no_event_outputs(env)


def test_scoring_record_with_another_model_revision_is_refused(tmp_path):
    env = scored_env(tmp_path)
    rec = json.loads(scoring_record_path(env).read_text(encoding="utf-8"))
    rec["model"]["model_revision"] = "7db323f79b751944bcfa66298ec06977e4518306"
    scoring_record_path(env).write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(ec.ClassificationError, match="model_revision"):
        classify(env)


def test_phase8_directories_are_refused(tmp_path):
    env = scored_env(tmp_path)
    with pytest.raises(ec.ClassificationError, match="Phase 8 sentiment directory"):
        ec.run_development_classification(report_path=env.report, sentiment_dir=historical_sentiment.SENTIMENT_DIR,
                                          output_dir=env.events, record_dir=env.records)
    with pytest.raises(ec.ClassificationError, match="Phase 8 events directory"):
        ec.run_development_classification(report_path=env.report, sentiment_dir=env.out,
                                          output_dir=ec.EVENTS_DIR_PHASE8, record_dir=env.records)


def test_score_file_with_a_missing_article_is_refused(tmp_path):
    env = scored_env(tmp_path)
    reseal_scores(env, lambda lines: lines[:-1])
    with pytest.raises(ec.ClassificationError, match="scoring run record fails verification|article set differs"):
        classify(env)
    assert no_event_outputs(env)


def test_score_file_with_a_non_development_record_is_refused(tmp_path):
    env = scored_env(tmp_path)

    def add_foreign(lines):
        rec = json.loads(lines[0])
        rec["provider_article_id"] = "999999"
        return lines + [json.dumps(rec, sort_keys=True)]
    reseal_scores(env, add_foreign)
    with pytest.raises(ec.ClassificationError, match="not a development article"):
        classify(env)
    assert no_event_outputs(env)


def test_score_record_with_wrong_text_hash_is_refused(tmp_path):
    env = scored_env(tmp_path)

    def wrong_hash(lines):
        rec = json.loads(lines[0])
        rec["input_text_hash"] = "0" * 64
        return [json.dumps(rec, sort_keys=True)] + lines[1:]
    reseal_scores(env, wrong_hash)
    with pytest.raises(ec.ClassificationError, match="rejected"):
        classify(env)
    assert no_event_outputs(env)


def test_score_record_with_other_provenance_is_refused(tmp_path):
    env = scored_env(tmp_path)

    def other_rev(lines):
        rec = json.loads(lines[0])
        rec["model_revision"] = "7db323f79b751944bcfa66298ec06977e4518306"
        return [json.dumps(rec, sort_keys=True)] + lines[1:]
    reseal_scores(env, other_rev)
    with pytest.raises(ec.ClassificationError, match="rejected"):
        classify(env)


# ============================================================
# Hash mismatches
# ============================================================


def test_tampered_score_file_is_refused(tmp_path):
    env = scored_env(tmp_path)
    score_path(env).write_bytes(score_path(env).read_bytes() + b"\n")
    with pytest.raises(ec.ClassificationError, match="score file SHA256 does not match"):
        classify(env)


def test_changed_integrity_report_is_refused(tmp_path):
    env = scored_env(tmp_path)
    report = json.loads(env.report.read_text(encoding="utf-8"))
    report["written_at"] = "later"
    env.report.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ec.ClassificationError, match="integrity_report.sha256"):
        classify(env)


def test_tampered_canonical_dataset_is_refused(tmp_path):
    env = scored_env(tmp_path)
    env.canonical.write_bytes(env.canonical.read_bytes().replace(b"Acme", b"Acmf", 1))
    with pytest.raises(ec.ScoringError, match="SHA256 does not match the integrity report"):
        classify(env)
    assert no_event_outputs(env)


def test_frozen_artifact_drift_is_refused(tmp_path, monkeypatch):
    import training.phase11b_sentiment_scoring as scoring
    env = scored_env(tmp_path)
    monkeypatch.setitem(scoring.FROZEN_ARTIFACTS, "amendment_04", (scoring.AMENDMENT_04_PATH, "0" * 64))
    with pytest.raises(ec.ScoringError, match="amendment_04"):
        classify(env)


# ============================================================
# Determinism, immutability, CLI, schema compatibility
# ============================================================


def test_nondeterministic_classification_is_refused(tmp_path, monkeypatch):
    env = scored_env(tmp_path)
    calls = []
    original = KeywordEventClassifier.classify_texts

    def flaky(self, texts):
        calls.append(1)
        out = original(self, texts)
        return out if len(calls) == 1 else list(reversed(out))
    monkeypatch.setattr(KeywordEventClassifier, "classify_texts", flaky)
    with pytest.raises(ec.ClassificationError, match="not deterministic"):
        classify(env)
    assert no_event_outputs(env)


def test_output_is_byte_identical_across_independent_runs(tmp_path):
    a, b = scored_env(tmp_path / "a"), scored_env(tmp_path / "b")
    classify(a)
    classify(b)
    assert events_path(a).read_bytes() == events_path(b).read_bytes()


def test_completed_run_is_write_once(tmp_path):
    env = scored_env(tmp_path)
    classify(env)
    before = events_path(env).read_bytes()
    with pytest.raises(ec.ClassificationError, match="run record already exists"):
        classify(env)
    assert events_path(env).read_bytes() == before


def test_existing_events_file_with_different_content_is_not_overwritten(tmp_path):
    env = scored_env(tmp_path)
    env.events.mkdir(parents=True)
    events_path(env).write_bytes(b"something else\n")
    with pytest.raises(ec.ClassificationError, match="exists with different content"):
        classify(env)
    assert events_path(env).read_bytes() == b"something else\n"
    assert not list(env.records.glob("rules_v1_development_classification_*.json"))


@pytest.mark.parametrize("extra", [["--canonical", "x.jsonl"], ["--start", "2017-01-01"], ["--include-sealed"]])
def test_cli_accepts_no_paths_dates_or_selection(extra):
    with pytest.raises(SystemExit) as exc:
        ec.main(extra)
    assert exc.value.code == 2


def test_outputs_keep_the_event_records_v1_schema(tmp_path):
    """Synthetic data only: the existing Phase 6 loader re-joins and re-classifies the stored labels."""
    env = scored_env(tmp_path)
    classify(env)
    classified, meta = load_event_dataset(events_path(env), score_path(env), env.canonical,
                                          verify_with=KeywordEventClassifier())
    assert len(classified) == 5 and meta["phase11b"]["rules_v1_spec_sha256"] == ec.RULES_V1_SPEC_SHA256
    assert meta["labels_are_ground_truth"] is False and not any(p.name.endswith(".tmp") for p in env.events.iterdir())
