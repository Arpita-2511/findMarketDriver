"""
FinBERT sentiment service (Phase 5A) with a deterministic FAKE backend.
Offline: no torch, no transformers, no model download.
"""

import hashlib
import math
from datetime import datetime, timezone

import numpy as np
import pytest

from services.finbert_sentiment import (
    INFERENCE_VERSION,
    LABELS,
    TEXT_POLICY,
    FinBertSentimentService,
    SentimentModelError,
    SentimentResult,
    SentimentTextError,
    build_sentiment_text,
    compose_sentiment_text,
    load_error,
    resolve_device,
    text_hash,
)
from services.news_schema import NewsArticle

T = datetime(2024, 7, 10, 14, 30, tzinfo=timezone.utc)


class FakeBackend:
    """Keyword-driven logits, unique per text (length term), columns in `labels` order."""

    model_name = "fake/finbert"
    model_revision = "0123456789abcdef"

    def __init__(self, labels=("positive", "negative", "neutral")):
        self.labels = labels
        self.calls = []

    def predict_logits(self, texts):
        self.calls.append(list(texts))
        rows = []
        for t in texts:
            low = t.lower()
            score = {
                "positive": 3.0 if any(w in low for w in ("beat", "surge", "record")) else 0.0,
                "negative": 3.0 if any(w in low for w in ("miss", "plunge", "lawsuit")) else 0.0,
                "neutral": 1.0 + (len(t) % 7) / 10,
            }
            rows.append([score[label] for label in self.labels])
        return np.array(rows)


class BrokenBackend(FakeBackend):
    def __init__(self, logits=None, exc=None, labels=("positive", "negative", "neutral")):
        super().__init__(labels)
        self._logits, self._exc = logits, exc

    def predict_logits(self, texts):
        if self._exc:
            raise self._exc
        return self._logits


def service(backend=None, **kwargs):
    return FinBertSentimentService(backend or FakeBackend(), **kwargs)


def article(**overrides):
    fields = dict(provider="alpaca", provider_article_id="42", headline="Apple beats estimates",
                  summary="Revenue rose 8%.", content="<p>Full <b>story</b></p>", symbols=("AAPL", "MSFT"),
                  source="benzinga", source_url="https://example.com/news/42",
                  created_at=T, information_available_at=T, fetched_at=T)
    fields.update(overrides)
    return NewsArticle(**fields)


# ==========================================
# Text policy
# ==========================================


def test_headline_plus_summary():
    assert compose_sentiment_text("Apple beats estimates", "Revenue rose 8%.") == "Apple beats estimates Revenue rose 8%."


def test_headline_only():
    assert compose_sentiment_text("Apple beats estimates", None) == "Apple beats estimates"
    assert compose_sentiment_text("Apple beats estimates", "   ") == "Apple beats estimates"


def test_summary_only():
    assert compose_sentiment_text("", "Revenue rose 8%.") == "Revenue rose 8%."


def test_empty_article_text_rejected():
    with pytest.raises(SentimentTextError):
        compose_sentiment_text("  ", "\n\t", None)
    with pytest.raises(SentimentTextError):
        compose_sentiment_text(None, None, "<p> </p>")


def test_whitespace_normalized():
    assert compose_sentiment_text("  Apple \n beats\t estimates ", " Revenue   rose ") == \
        "Apple beats estimates Revenue rose"


def test_summary_repeating_headline_is_dropped():
    assert compose_sentiment_text("Apple Beats Estimates", "apple beats  estimates") == "Apple Beats Estimates"


def test_content_used_only_when_headline_and_summary_empty():
    assert compose_sentiment_text("", "", "<p>Shares &amp; bonds <b>fell</b></p>") == "Shares & bonds fell"
    assert compose_sentiment_text("Headline", "", "<p>ignored body</p>") == "Headline"
    assert compose_sentiment_text("", "", "<p>&amp;lt;</p>") == "&lt;"          # entities decoded once


def test_article_text_excludes_metadata_and_does_not_mutate():
    a = article()
    before = a.to_dict()
    text = build_sentiment_text(a)
    assert text == "Apple beats estimates Revenue rose 8%."
    for leaked in ("https://", "example.com", "MSFT", "benzinga", "42", "2024", "Full story"):
        assert leaked not in text
    assert a.to_dict() == before


# ==========================================
# Probabilities, label, score, schema
# ==========================================


@pytest.mark.parametrize("text, label", [
    ("Apple beats estimates with record revenue", "positive"),
    ("Apple shares plunge after earnings miss", "negative"),
    ("Apple schedules annual meeting", "neutral"),
])
def test_probabilities_label_and_score(text, label):
    r = service().score_text(text)
    probs = r.probabilities()
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-12)
    assert r.label == label == max(probs, key=probs.get)
    assert r.sentiment_score == pytest.approx(r.positive_probability - r.negative_probability)
    assert -1.0 <= r.sentiment_score <= 1.0


def test_backend_label_order_is_respected():
    """FinBERT's own id2label order may differ; results must not depend on column order."""
    text = "Apple beats estimates"
    default = service().score_text(text)
    shuffled = service(FakeBackend(labels=("neutral", "positive", "negative"))).score_text(text)
    assert shuffled.probabilities() == pytest.approx(default.probabilities())
    assert shuffled.label == default.label


def test_ties_break_in_fixed_label_order():
    r = service(BrokenBackend(logits=np.zeros((1, 3)))).score_text("flat")
    assert r.label == "positive" and r.sentiment_score == 0.0


def test_output_schema_and_provenance():
    text = "Apple beats estimates"
    r = service().score_text(text)
    assert set(r.to_dict()) == {"inference_version", "model_name", "model_revision", "label",
                                "positive_probability", "negative_probability", "neutral_probability",
                                "sentiment_score", "input_text_hash", "text_policy"}
    assert r.inference_version == INFERENCE_VERSION
    assert (r.model_name, r.model_revision) == ("fake/finbert", "0123456789abcdef")
    assert r.input_text_hash == hashlib.sha256(text.encode("utf-8")).hexdigest() == text_hash(text)
    assert r.text_policy == TEXT_POLICY


def test_article_scoring_uses_text_policy():
    a = article()
    r = service().score_article(a)
    assert r.input_text_hash == text_hash(build_sentiment_text(a))


# ==========================================
# Determinism, ordering, batching
# ==========================================


def test_same_input_same_output():
    s = service()
    assert s.score_text("Apple beats estimates") == s.score_text("Apple beats estimates")


def test_batch_matches_individual_scoring_and_keeps_order():
    texts = ["Apple beats", "Apple plunge", "Apple meeting", "Record surge", "Lawsuit filed"]
    backend = FakeBackend()
    batched = service(backend, batch_size=2).score_texts(texts)
    single = [service().score_text(t) for t in texts]

    assert batched == single                                     # no cross-text mixing, same order
    assert [len(c) for c in backend.calls] == [2, 2, 1]          # configurable batch size
    assert [r.input_text_hash for r in batched] == [text_hash(t) for t in texts]


def test_empty_batch():
    backend = FakeBackend()
    assert service(backend).score_texts([]) == []
    assert service(backend).score_articles([]) == []
    assert backend.calls == []


def test_empty_text_in_batch_rejected_before_inference():
    backend = FakeBackend()
    with pytest.raises(SentimentTextError, match="#1"):
        service(backend).score_texts(["ok", "   "])
    assert backend.calls == []


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        service(batch_size=0)


# ==========================================
# Malformed model output
# ==========================================


@pytest.mark.parametrize("logits, message", [
    (np.zeros((2, 3)), "shape"),                     # wrong row count for one text
    (np.zeros((1, 2)), "shape"),                     # missing class
    (np.array([[np.nan, 0.0, 0.0]]), "non-finite"),
    (np.array([[np.inf, 0.0, 0.0]]), "non-finite"),
])
def test_malformed_logits_raise(logits, message):
    with pytest.raises(SentimentModelError, match=message):
        service(BrokenBackend(logits=logits)).score_text("text")


def test_backend_with_wrong_labels_rejected():
    with pytest.raises(SentimentModelError, match="labels"):
        service(FakeBackend(labels=("up", "down", "flat")))


def test_backend_exception_becomes_model_error():
    with pytest.raises(SentimentModelError, match="RuntimeError"):
        service(BrokenBackend(exc=RuntimeError("tensor size mismatch"))).score_text("text")


@pytest.mark.parametrize("overrides, message", [
    ({"positive_probability": 0.7}, "sum"),                                   # 0.7 + 0.2 + 0.3
    ({"label": "neutral"}, "most probable"),
    ({"sentiment_score": 0.9}, "sentiment_score"),
    ({"positive_probability": 1.2, "negative_probability": -0.4}, "in \\[0, 1\\]"),
    ({"label": "bullish"}, "unknown label"),
])
def test_result_rejects_inconsistent_values(overrides, message):
    fields = dict(inference_version=INFERENCE_VERSION, model_name="m", model_revision="r", label="positive",
                  positive_probability=0.5, negative_probability=0.2, neutral_probability=0.3,
                  sentiment_score=0.3, input_text_hash="0" * 64)
    fields.update(overrides)
    with pytest.raises(SentimentModelError, match=message):
        SentimentResult(**fields)


# ==========================================
# Device & secrets
# ==========================================


def test_device_resolution_defaults_to_cpu_and_never_assumes_cuda():
    assert resolve_device() == "cpu"
    assert resolve_device(" CPU ") == "cpu"
    assert resolve_device("cuda", cuda_available=lambda: True) == "cuda"
    with pytest.raises(SentimentModelError, match="CUDA is not available"):
        resolve_device("cuda", cuda_available=lambda: False)
    with pytest.raises(SentimentModelError, match="unsupported device"):
        resolve_device("tpu")


def test_secrets_are_redacted_from_errors(monkeypatch):
    token = "hf_FAKE_token_value_123456"
    monkeypatch.setenv("HF_TOKEN", token)
    assert token not in str(load_error("ProsusAI/finbert", OSError(f"401 Unauthorized for {token}")))
    with pytest.raises(SentimentModelError) as exc:
        service(BrokenBackend(exc=RuntimeError(f"bad header {token}"))).score_text("text")
    assert token not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__


def test_labels_constant():
    assert LABELS == ("positive", "negative", "neutral")
