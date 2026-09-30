"""
REAL ProsusAI/finbert smoke test (Phase 5A). Opt-in:

    pytest -m integration tests/integration/test_live_finbert.py -s

First run downloads the model into the Hugging Face cache (~440 MB, needs
internet); later runs use the cache. Nothing is written to the repository.
Skipped if torch / transformers are not installed.
"""

import time

import pytest

from services.finbert_sentiment import LABELS, FinBertSentimentService

pytestmark = pytest.mark.integration

POSITIVE = "Apple reports record quarterly revenue and raises its dividend, beating analyst expectations."
NEGATIVE = "Apple shares plunge after the company misses earnings estimates and cuts its outlook."
NEUTRAL = "Apple will hold its annual shareholder meeting on Friday."


@pytest.fixture(scope="module")
def finbert():
    # checked here, not at import time, so the default (deselected) run never reports a skip
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    started = time.perf_counter()
    svc = FinBertSentimentService.load(device="cpu")
    print(f"\nmodel load: {time.perf_counter() - started:.1f}s  revision={svc.backend.model_revision}")
    return svc


def test_model_and_tokenizer_load(finbert):
    backend = finbert.backend
    assert backend.model_name == "ProsusAI/finbert"
    assert backend.device == "cpu"
    assert sorted(backend.labels) == sorted(LABELS)
    assert backend.max_length == 512
    assert isinstance(backend.model_revision, str) and backend.model_revision


def test_valid_outputs_and_sensible_ordering(finbert):
    started = time.perf_counter()
    pos, neg, neu = finbert.score_texts([POSITIVE, NEGATIVE, NEUTRAL])
    print(f"3-sentence batch: {time.perf_counter() - started:.2f}s  "
          f"labels={pos.label}/{neg.label}/{neu.label}")
    for r in (pos, neg, neu):
        assert r.label in LABELS
        assert abs(sum(r.probabilities().values()) - 1.0) < 1e-6
        assert -1.0 <= r.sentiment_score <= 1.0
    assert pos.sentiment_score > neg.sentiment_score


def test_deterministic_and_batch_consistent(finbert):
    a, b = finbert.score_text(POSITIVE), finbert.score_text(POSITIVE)
    assert a.label == b.label
    assert a.probabilities() == pytest.approx(b.probabilities(), abs=1e-6)

    batched = finbert.score_texts([NEUTRAL, POSITIVE])                 # padded batch
    assert batched[1].label == a.label
    assert batched[1].probabilities() == pytest.approx(a.probabilities(), abs=1e-4)
