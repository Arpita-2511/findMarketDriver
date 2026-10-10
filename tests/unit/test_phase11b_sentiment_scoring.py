"""
Phase 11B Stage 11B.2: development-only FinBERT scoring - synthetic tests.

Fake FinBERT backend only; no real model, no real article, no network.
The real canonical dataset and real sealed records are never touched.
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest

import training.phase11b_sentiment_scoring as m
from services.finbert_sentiment import (
    INFERENCE_VERSION,
    FinBertSentimentService,
    SentimentModelError,
    SentimentResult,
    build_sentiment_text,
    text_hash,
)
from services.historical_sentiment import SENTIMENT_DIR, HistoricalSentimentError, load_sentiment_dataset, serialize_records
from services.news_schema import NewsArticle
from services.sentiment_features import ScoredArticle
from tests.sentiment_fakes import FakeFinBert, news
from training.phase11b_ingest_closeout import _development_article_sha256, _sealed_article_id_sha256

CUT = m.DEVELOPMENT_CUTOFF_UTC
H = timedelta(hours=1)
US = timedelta(microseconds=1)
SENTINEL = "SEALED_SENTINEL_TEXT_must_never_leave_the_partition"


# ============================================================
# Builders
# ============================================================


def _as_dict(x):
    return x.to_dict() if isinstance(x, NewsArticle) else dict(x)


def make_env(tmp_path, dev=(), sealed=(), *, raw_lines=None, report_overrides=None):
    """Synthetic canonical dataset (+ .meta.json) and a Gate B style integrity report computed independently."""
    dicts = [_as_dict(x) for x in (*dev, *sealed)]
    lines = raw_lines if raw_lines is not None else [json.dumps(d, sort_keys=True) for d in dicts]
    payload = "".join(line + "\n" for line in lines).encode("utf-8")
    canonical = tmp_path / "canonical.jsonl"
    canonical.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    canonical.with_suffix(".meta.json").write_text(json.dumps({"sha256": sha, "run_id": "synthetic"}), encoding="utf-8")
    dev_articles = [NewsArticle.from_dict(_as_dict(x)) for x in dev] if raw_lines is None else []
    sealed_keys = [f"{d['provider']}:{d['provider_article_id']}" for d in map(_as_dict, sealed)]
    report = {
        "split_applied_before_scoring": True,
        "development_sealed_intersection_empty": True,
        "development_cutoff_utc": "2024-09-30T20:30:00Z",
        "boundary_rule": m.BOUNDARY_RULE,
        "malformed_policy_amendment_sha256": m.MALFORMED_POLICY_AMENDMENT_SHA256,
        "run_cap_policy_amendment_sha256": m.RUN_CAP_POLICY_AMENDMENT_SHA256,
        "run_cap_evaluation": "final_totals",
        "development_article_count": len(dev_articles),
        "sealed_article_count": len(sealed_keys),
        "development_article_sha256": _development_article_sha256(dev_articles),
        "sealed_article_id_sha256": _sealed_article_id_sha256(sealed_keys),
        "canonical_input": {"file": canonical.name, "sha256": sha, "rows": len(lines), "run_id": "synthetic"},
        **(report_overrides or {}),
    }
    report_path = tmp_path / "news_integrity_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return SimpleNamespace(canonical=canonical, report=report_path, sha=sha, out=tmp_path / "sentiment" / "phase11b",
                           records=tmp_path / "records", run_id=m.run_id_for(sha))


def fake(revision=m.APPROVED_REVISION, batch_size=16, **kw):
    return FinBertSentimentService(FakeFinBert(revision=revision, **kw), batch_size=batch_size)


def run(env, service=None, *, factory_calls=None, **kw):
    service = service or fake()

    def factory():
        if factory_calls is not None:
            factory_calls.append(1)
        return service
    return m.run_development_scoring(env.canonical, factory, report_path=env.report, output_dir=env.out,
                                     record_dir=env.records, **kw)


def texts(service):
    return [t for call in service.backend.calls for t in call]


def no_outputs(env):
    scores = list(env.out.glob("*.sentiment.jsonl")) if env.out.exists() else []
    records = list(env.records.glob("*.json")) if env.records.exists() else []
    return not scores and not records


def scored_ids(env):
    path = env.out / f"{env.run_id}.sentiment.jsonl"
    return [json.loads(line)["provider_article_id"] for line in path.read_text(encoding="utf-8").splitlines()]


class FlakyFinBert(FakeFinBert):
    """Raises on the n-th inference call (simulated interruption)."""

    def __init__(self, fail_on_call, **kw):
        super().__init__(**kw)
        self.fail_on_call = fail_on_call

    def predict_logits(self, batch):
        if len(self.calls) + 1 == self.fail_on_call:
            raise RuntimeError("simulated interruption")
        return super().predict_logits(batch)


def foreign_checkpoint_bytes(i):
    """A checkpoint record with the approved provenance but for an article outside the development set."""
    a = news(i, CUT - 5 * H)
    r = SentimentResult(inference_version=INFERENCE_VERSION, model_name="ProsusAI/finbert",
                        model_revision=m.APPROVED_REVISION, label="neutral", positive_probability=0.2,
                        negative_probability=0.1, neutral_probability=0.7, sentiment_score=0.2 - 0.1,
                        input_text_hash=text_hash(build_sentiment_text(a)))
    return serialize_records([ScoredArticle(a, r)])


# ============================================================
# 1-5. Partition: cutoff, availability rule, time zones
# ============================================================


def test_development_article_selected_and_only_it_is_scored(tmp_path):
    env = make_env(tmp_path, dev=[news(1, CUT - H)], sealed=[news(2, CUT + H)])
    service = fake()
    record = run(env, service)
    assert (record["development_article_count"], record["sealed_article_count"]) == (1, 1)
    assert (record["eligible_article_count"], record["scored_article_count"], record["excluded_article_count"]) == (1, 1, 0)
    assert scored_ids(env) == ["1"] and len(texts(service)) == 1


def test_article_available_exactly_at_cutoff_is_development(tmp_path):
    """Frozen rule: information_available_at <= 2024-09-30T20:30:00Z is development."""
    env = make_env(tmp_path, dev=[news(1, CUT)])
    assert run(env)["development_article_count"] == 1 and scored_ids(env) == ["1"]


def test_article_one_microsecond_after_cutoff_is_sealed_and_not_scored(tmp_path):
    env = make_env(tmp_path, dev=[news(1, CUT)], sealed=[news(2, CUT + US)])
    service = fake()
    record = run(env, service)
    assert record["sealed_article_count"] == 1 and scored_ids(env) == ["1"] and len(texts(service)) == 1


@pytest.mark.parametrize("created, updated, side", [
    (CUT - 48 * H, CUT + H, "sealed"),         # revised after the cutoff -> available after it
    (CUT - 48 * H, CUT - 24 * H, "dev"),       # revised before the cutoff
    (CUT - H, CUT - 3 * H, "dev"),             # updated earlier than created -> created governs
    (CUT + H, CUT - 3 * H, "sealed"),          # created after the cutoff, earlier update ignored
])
def test_availability_is_max_of_created_and_updated(tmp_path, created, updated, side):
    item = news(2, created, updated=updated)
    env = make_env(tmp_path, dev=[news(1, CUT - 2 * H)] + ([item] if side == "dev" else []),
                   sealed=[item] if side == "sealed" else [])
    run(env)
    assert ("2" in scored_ids(env)) == (side == "dev")


def test_timestamp_offsets_are_compared_as_instants(tmp_path):
    at_cut = news(1, CUT).to_dict()
    at_cut.update(created_at="2024-09-30T16:30:00-04:00", information_available_at="2024-09-30T16:30:00-04:00")
    after = news(2, CUT + US).to_dict()
    after.update(created_at="2024-10-01T01:30:00.000001+05:00", information_available_at="2024-10-01T01:30:00.000001+05:00")
    env = make_env(tmp_path, dev=[at_cut], sealed=[after])
    record = run(env)
    assert (record["development_article_count"], record["sealed_article_count"]) == (1, 1) and scored_ids(env) == ["1"]


@pytest.mark.parametrize("side", ["dev", "sealed"])
@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.pop("created_at"), "created_at"),
    (lambda d: d.update(created_at="not-a-date"), "created_at"),
    (lambda d: d.update(created_at="2024-09-29T10:00:00", information_available_at="2024-09-29T10:00:00"), "timezone"),
    (lambda d: d.update(information_available_at="2024-09-29T00:00:00+00:00"), "not max"),
])
def test_missing_or_malformed_timestamps_fail_closed(tmp_path, side, mutate, message):
    bad = news(2, CUT - H if side == "dev" else CUT + H).to_dict()
    mutate(bad)
    env = make_env(tmp_path, raw_lines=[json.dumps(news(1, CUT - 2 * H).to_dict()), json.dumps(bad)])
    calls = []
    with pytest.raises(m.ScoringError, match=message):
        run(env, factory_calls=calls)
    assert calls == [] and no_outputs(env)                              # model never loaded, nothing written


# ============================================================
# 7-8. Sealed content never read; development interface is closed
# ============================================================


def test_sealed_content_is_never_parsed_scored_logged_or_reported(tmp_path, monkeypatch, capsys):
    sealed = news(9, CUT + H).to_dict()
    sealed.update(headline=123, summary=SENTINEL, content=SENTINEL)     # a full parse of this record would raise
    env = make_env(tmp_path, dev=[news(1, CUT - H), news(2, CUT)], sealed=[sealed])
    parsed = []
    original = NewsArticle.from_dict
    monkeypatch.setattr(NewsArticle, "from_dict",
                        classmethod(lambda cls, d: (parsed.append(d.get("provider_article_id")), original(d))[1]))
    service = fake()
    record = run(env, service)
    assert record["sealed_article_count"] == 1 and record["sealed_content_sent_to_scoring"] is False
    assert "9" not in parsed and sorted(parsed) == ["1", "2"]
    assert all(SENTINEL not in t for t in texts(service)) and len(texts(service)) == 2
    for path in [*env.out.iterdir(), *env.records.iterdir()]:
        assert SENTINEL.encode() not in path.read_bytes(), path.name
    assert SENTINEL not in capsys.readouterr().out


def test_development_interface_rejects_sealed_or_unlisted_articles_before_text_is_built():
    service = fake()
    sealed = news(9, CUT + US)
    with pytest.raises(m.ScoringError, match="not a development article"):
        m.score_development_subset([sealed], {sealed.article_key}, service, checkpoint=None)
    unlisted = news(8, CUT - H)
    with pytest.raises(m.ScoringError, match="not a development article"):
        m.score_development_subset([unlisted], set(), service, checkpoint=None)
    assert service.backend.calls == []


@pytest.mark.parametrize("extra", [["--end", "2026-09-29"], ["--start", "2017-01-01"], ["--cutoff", "2026-09-29"],
                                   ["--article-ids", "1"], ["--include-sealed"]])
def test_cli_accepts_no_date_range_cutoff_or_article_selection(extra, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(SystemExit) as exc:
        m.main(["--canonical", "does-not-exist.jsonl", *extra])
    assert exc.value.code == 2


def test_cli_requires_hf_hub_offline(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(SystemExit, match="HF_HUB_OFFLINE=1 is required"):
        m.main(["--canonical", "does-not-exist.jsonl"])


# ============================================================
# 9, 13. Phase 8 reuse, checkpoint binding, resume
# ============================================================


def test_output_dir_cannot_be_the_phase8_sentiment_directory(tmp_path):
    env = make_env(tmp_path, dev=[news(1, CUT - H)])
    with pytest.raises(m.ScoringError, match="Phase 8 sentiment directory"):
        m.run_development_scoring(env.canonical, fake, report_path=env.report, output_dir=SENTIMENT_DIR,
                                  record_dir=env.records)


def test_checkpoint_without_binding_is_refused(tmp_path):
    env = make_env(tmp_path, dev=[news(1, CUT - H)])
    env.out.mkdir(parents=True)
    (env.out / f"{env.run_id}.checkpoint.jsonl").write_bytes(foreign_checkpoint_bytes(1))   # e.g. a copied Phase 8 file
    service = fake()
    with pytest.raises(m.ScoringError, match="without a binding"):
        run(env, service)
    assert service.backend.calls == [] and no_outputs(env)


def _interrupted(tmp_path, n_dev=520):
    env = make_env(tmp_path, dev=[news(i, CUT - (i + 1) * timedelta(minutes=1)) for i in range(1, n_dev + 1)])
    flaky = FinBertSentimentService(FlakyFinBert(2, revision=m.APPROVED_REVISION), batch_size=512)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(env, flaky)
    checkpoint = env.out / f"{env.run_id}.checkpoint.jsonl"
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 512 and no_outputs(env)
    return env, checkpoint


def test_interrupted_run_resumes_from_its_bound_checkpoint(tmp_path):
    env, _ = _interrupted(tmp_path)
    service = fake(batch_size=512)
    record = run(env, service)
    assert record["scored_from_checkpoint_count"] == 512 and record["scored_article_count"] == 520
    assert len(texts(service)) == 8                                     # only the unscored remainder


def test_checkpoint_binding_mismatch_is_refused(tmp_path):
    env, _ = _interrupted(tmp_path)
    binding_path = env.out / f"{env.run_id}.checkpoint.binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["development_article_sha256"] = "0" * 64
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(m.ScoringError, match="binding does not match"):
        run(env, fake(batch_size=512))
    assert no_outputs(env)


def test_checkpoint_with_non_development_records_is_refused(tmp_path):
    env, checkpoint = _interrupted(tmp_path)
    with open(checkpoint, "ab") as f:
        f.write(foreign_checkpoint_bytes(999_999))
    service = fake(batch_size=512)
    with pytest.raises(m.ScoringError, match="non-development record"):
        run(env, service)
    assert service.backend.calls == [] and no_outputs(env)


# ============================================================
# 10-12. Model identity, model output, empty input, duplicates, missing text
# ============================================================


@pytest.mark.parametrize("service, message", [
    (lambda: fake(revision="7db323f79b751944bcfa66298ec06977e4518306"), "revision"),
    (lambda: fake(revision=None), "revision"),
])
def test_unapproved_model_is_refused_before_scoring(tmp_path, service, message):
    env = make_env(tmp_path, dev=[news(1, CUT - H)])
    svc = service()
    with pytest.raises(m.ScoringError, match=message):
        run(env, svc)
    assert svc.backend.calls == [] and no_outputs(env)


def test_unexpected_model_classes_are_refused_when_the_model_loads(tmp_path):
    env = make_env(tmp_path, dev=[news(1, CUT - H)])
    with pytest.raises(SentimentModelError, match="labels"):
        m.run_development_scoring(env.canonical, lambda: fake(labels=("pos", "neg", "neu")), report_path=env.report,
                                  output_dir=env.out, record_dir=env.records)
    assert no_outputs(env)


def test_wrong_model_name_is_refused(tmp_path):
    env = make_env(tmp_path, dev=[news(1, CUT - H)])
    svc = fake()
    svc.backend.model_name = "someone/other-finbert"
    with pytest.raises(m.ScoringError, match="not the approved"):
        run(env, svc)
    assert no_outputs(env)


class NanFinBert(FakeFinBert):
    def predict_logits(self, batch):
        super().predict_logits(batch)
        return np.full((len(batch), 3), np.nan)


class ShortFinBert(FakeFinBert):
    def predict_logits(self, batch):
        return super().predict_logits(batch)[:-1]


@pytest.mark.parametrize("backend", [NanFinBert, ShortFinBert])
def test_invalid_model_output_fails_without_writing(tmp_path, backend):
    env = make_env(tmp_path, dev=[news(1, CUT - H), news(2, CUT - 2 * H)])
    with pytest.raises((SentimentModelError, HistoricalSentimentError, m.ScoringError)):
        run(env, FinBertSentimentService(backend(revision=m.APPROVED_REVISION)))
    assert no_outputs(env)


def test_empty_development_set_fails_before_loading_the_model(tmp_path):
    env = make_env(tmp_path, sealed=[news(1, CUT + H)])
    calls = []
    with pytest.raises(m.ScoringError, match="no development articles"):
        run(env, factory_calls=calls)
    assert calls == [] and no_outputs(env)


def test_duplicate_identifiers_fail_closed(tmp_path):
    line = json.dumps(news(1, CUT - H).to_dict(), sort_keys=True)
    env = make_env(tmp_path, raw_lines=[line, line])
    with pytest.raises(m.ScoringError, match="duplicate article identifier"):
        run(env)


@pytest.mark.parametrize("mutate", [lambda d: d.update(headline=""), lambda d: d.update(provider_article_id=" ")])
def test_missing_text_or_identifier_fails_closed(tmp_path, mutate):
    bad = news(2, CUT - H).to_dict()
    mutate(bad)
    env = make_env(tmp_path, raw_lines=[json.dumps(bad)])
    with pytest.raises(m.ScoringError, match="invalid"):
        run(env)
    assert no_outputs(env)


# ============================================================
# 14. Outputs, provenance, write-once, no secrets
# ============================================================


def test_outputs_record_and_compatibility(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_FAKE_SECRET_value_123456")
    monkeypatch.setenv("ALPACA_API_SECRET", "ALPACA_FAKE_SECRET_7890")
    env = make_env(tmp_path, dev=[news(1, CUT - H), news(2, CUT - 2 * H)], sealed=[news(3, CUT + H)])
    record = run(env)
    scores = env.out / f"{env.run_id}.sentiment.jsonl"
    assert record["outputs"]["scores"]["sha256"] == hashlib.sha256(scores.read_bytes()).hexdigest()
    assert record["model"]["model_revision"] == m.APPROVED_REVISION and record["model"]["model_name"] == "ProsusAI/finbert"
    assert record["frozen_artifact_sha256"] == {k: v[1] for k, v in m.FROZEN_ARTIFACTS.items()}
    assert record["phase_8_score_reuse"] is False and record["partition_reproduces_integrity_report"] is True
    assert record["integrity_report"]["sha256"] == hashlib.sha256(env.report.read_bytes()).hexdigest()
    assert record["development_cutoff_utc"] == "2024-09-30T20:30:00Z" and record["run_id"] == env.run_id
    scored, meta = load_sentiment_dataset(scores, env.canonical)          # existing Phase 5C/8 loader accepts it
    assert [s.article.provider_article_id for s in scored] == ["2", "1"] and meta["phase11b"]["stage"] == "11B.2"
    for path in [*env.out.iterdir(), *env.records.iterdir()]:
        body = path.read_bytes()
        assert b"FAKE_SECRET" not in body, path.name
        assert not path.name.endswith(".tmp")


def test_completed_run_is_write_once(tmp_path):
    env = make_env(tmp_path, dev=[news(1, CUT - H)])
    run(env)
    service = fake()
    with pytest.raises(m.ScoringError, match="run record already exists"):
        run(env, service)
    assert service.backend.calls == []


# ============================================================
# 15. Frozen artefacts, Gate B facts, partition reproduction
# ============================================================


def test_frozen_artifact_drift_is_refused(tmp_path, monkeypatch):
    env = make_env(tmp_path, dev=[news(1, CUT - H)])
    monkeypatch.setitem(m.FROZEN_ARTIFACTS, "amendment_05", (m.AMENDMENT_05_PATH, "0" * 64))
    with pytest.raises(m.ScoringError, match="amendment_05"):
        run(env)


def test_real_frozen_artifacts_verify():
    assert m.verify_frozen_artifacts()["design_freeze"] == "cf7a14fbce45dc466a66993ffb985f4623022bb33a320caae0d6947495436d5f"


@pytest.mark.parametrize("overrides", [
    {"boundary_rule": "information_available_at < cutoff -> development; >= cutoff -> sealed"},
    {"split_applied_before_scoring": False},
    {"development_cutoff_utc": "2024-09-30T20:30:01Z"},
    {"run_cap_evaluation": "per_chunk"},
    {"malformed_policy_amendment_sha256": "0" * 64},
])
def test_integrity_report_must_carry_gate_b_facts(tmp_path, overrides):
    env = make_env(tmp_path, dev=[news(1, CUT - H)], report_overrides=overrides)
    calls = []
    with pytest.raises(m.ScoringError, match="integrity report fails"):
        run(env, factory_calls=calls)
    assert calls == []


@pytest.mark.parametrize("overrides", [
    {"development_article_count": 2}, {"sealed_article_count": 0},
    {"development_article_sha256": "0" * 64}, {"sealed_article_id_sha256": "0" * 64},
])
def test_partition_must_reproduce_the_integrity_report(tmp_path, overrides):
    env = make_env(tmp_path, dev=[news(1, CUT - H)], sealed=[news(2, CUT + H)], report_overrides=overrides)
    calls = []
    with pytest.raises(m.ScoringError, match="does not reproduce the integrity report"):
        run(env, factory_calls=calls)
    assert calls == [] and no_outputs(env)


def test_tampered_canonical_dataset_is_refused_before_loading_the_model(tmp_path):
    env = make_env(tmp_path, dev=[news(1, CUT - H)])
    env.canonical.write_bytes(env.canonical.read_bytes().replace(b"Apple update 1", b"Apple update X"))
    calls = []
    with pytest.raises(m.ScoringError, match="SHA256 does not match the integrity report"):
        run(env, factory_calls=calls)
    assert calls == [] and no_outputs(env)
