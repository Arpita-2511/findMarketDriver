"""
Article-level financial sentiment with pretrained FinBERT (Phase 5A).

Model: ProsusAI/finbert (Hugging Face), inference only - no fine-tuning.
Weights are downloaded into the normal Hugging Face cache on first use and
are never stored in this repository.

Layers:

    compose_sentiment_text / build_sentiment_text   pure text policy (TEXT_POLICY)
    SentimentBackend                                model boundary: texts -> raw logits
        TransformersFinBertBackend                  real model (torch/transformers,
                                                    imported lazily, loaded ONCE)
    FinBertSentimentService                         batching, softmax, validation,
                                                    SentimentResult

The service never persists anything and never touches NewsArticle
timestamps: sentiment is an attribute of an article's text; WHEN it may be
used is still decided only by information_available_at <= prediction_timestamp
(services/news_alignment.py).

Unit tests use a fake backend (no torch, no network). The real model is
exercised only by the opt-in integration test.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Iterable, Protocol, Sequence

import numpy as np

from services.news_schema import NewsArticle

FINBERT_MODEL = "ProsusAI/finbert"
INFERENCE_VERSION = "finbert_sentiment_v1"
LABELS = ("positive", "negative", "neutral")      # fixed order: also the argmax tie-break order
PROBABILITY_TOLERANCE = 1e-6
DEFAULT_BATCH_SIZE = 16

TEXT_POLICY = ("text_v1: headline + ' ' + summary (whitespace-collapsed; summary dropped if it "
               "repeats the headline); content (HTML stripped) only if both are empty; "
               "never URLs, ids, symbols, timestamps or source")

# Environment variables whose VALUES must never appear in error messages
SECRET_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ALPACA_API_KEY", "ALPACA_API_SECRET")


class SentimentTextError(ValueError):
    """No usable text to score."""


class SentimentModelError(RuntimeError):
    """Model could not be loaded, or returned structurally invalid output."""


# ==========================================
# Text policy
# ==========================================

_WHITESPACE = re.compile(r"\s+")


def normalize_whitespace(text: str | None) -> str:
    return _WHITESPACE.sub(" ", text).strip() if text else ""


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data):
        self.parts.append(data)


def html_to_text(html: str | None) -> str:
    """Visible text of an HTML fragment (tags removed, entities decoded once), whitespace-collapsed."""
    if not html:
        return ""
    parser = _TextExtractor()          # convert_charrefs=True decodes entities exactly once
    parser.feed(html)
    parser.close()
    return normalize_whitespace(" ".join(parser.parts))


def compose_sentiment_text(headline: str | None, summary: str | None, content: str | None = None) -> str:
    """Apply TEXT_POLICY to raw fields. Raises SentimentTextError if nothing usable remains."""
    head = normalize_whitespace(headline)
    summ = normalize_whitespace(summary)
    if summ and summ.casefold() == head.casefold():
        summ = ""
    text = " ".join(part for part in (head, summ) if part)
    if not text:
        text = html_to_text(content)
    if not text:
        raise SentimentTextError("article has no usable headline, summary or content text")
    return text


def build_sentiment_text(article: NewsArticle) -> str:
    """TEXT_POLICY for a canonical article (read-only; the article is not modified)."""
    return compose_sentiment_text(article.headline, article.summary, article.content)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ==========================================
# Result schema
# ==========================================


@dataclass(frozen=True)
class SentimentResult:
    inference_version: str
    model_name: str
    model_revision: str | None
    label: str
    positive_probability: float
    negative_probability: float
    neutral_probability: float
    sentiment_score: float            # positive_probability - negative_probability, in [-1, 1]
    input_text_hash: str              # sha256 of the exact model input text
    text_policy: str = TEXT_POLICY

    def __post_init__(self):
        probs = self.probabilities()
        for name, p in probs.items():
            if not (isinstance(p, float) and math.isfinite(p) and 0.0 <= p <= 1.0):
                raise SentimentModelError(f"{name} probability {p!r} is not a finite value in [0, 1]")
        if abs(sum(probs.values()) - 1.0) > PROBABILITY_TOLERANCE:
            raise SentimentModelError(f"probabilities sum to {sum(probs.values())!r}, not 1")
        if self.label not in LABELS:
            raise SentimentModelError(f"unknown label {self.label!r}")
        if self.label != max(LABELS, key=lambda lbl: (probs[lbl], -LABELS.index(lbl))):
            raise SentimentModelError(f"label {self.label!r} is not the most probable class")
        expected = self.positive_probability - self.negative_probability
        if abs(self.sentiment_score - expected) > PROBABILITY_TOLERANCE:
            raise SentimentModelError("sentiment_score must equal positive - negative probability")
        if not re.fullmatch(r"[0-9a-f]{64}", self.input_text_hash):
            raise SentimentModelError("input_text_hash must be a sha256 hex digest")

    def probabilities(self) -> dict[str, float]:
        return {"positive": self.positive_probability, "negative": self.negative_probability,
                "neutral": self.neutral_probability}

    def to_dict(self) -> dict:
        return {
            "inference_version": self.inference_version,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "label": self.label,
            "positive_probability": self.positive_probability,
            "negative_probability": self.negative_probability,
            "neutral_probability": self.neutral_probability,
            "sentiment_score": self.sentiment_score,
            "input_text_hash": self.input_text_hash,
            "text_policy": self.text_policy,
        }


# ==========================================
# Model boundary
# ==========================================


class SentimentBackend(Protocol):
    model_name: str
    model_revision: str | None
    labels: Sequence[str]        # label of each logit column, e.g. ("positive", "negative", "neutral")

    def predict_logits(self, texts: Sequence[str]) -> np.ndarray:
        """Raw logits, shape (len(texts), len(labels)), in input order."""


def redact_secrets(text: str) -> str:
    """Remove the values of known secret environment variables from a message."""
    for name in SECRET_ENV_VARS:
        value = os.environ.get(name)
        if value and len(value) >= 4:
            text = text.replace(value, "<redacted>")
    return text


def load_error(model_name: str, exc: BaseException) -> SentimentModelError:
    return SentimentModelError(
        f"could not load {model_name} ({type(exc).__name__}): {redact_secrets(str(exc))[:300]}"
    )


def resolve_device(requested: str = "cpu", *, cuda_available=None) -> str:
    """'cpu' (default) or 'cuda'. CUDA is optional and never assumed."""
    requested = (requested or "cpu").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if cuda_available is None:
            import torch
            cuda_available = torch.cuda.is_available
        if not cuda_available():
            raise SentimentModelError("device 'cuda' requested but CUDA is not available; use 'cpu'")
        return "cuda"
    raise SentimentModelError(f"unsupported device {requested!r}; use 'cpu' or 'cuda'")


class TransformersFinBertBackend:
    """
    Real FinBERT via transformers + torch. Loaded once; eval mode; no gradients.

    Tokenization: truncation by TOKENS at max_length = the model's own limit
    (min of tokenizer.model_max_length and max_position_embeddings; 512 for
    ProsusAI/finbert). Characters are never cut. A single text is not
    padded; a batch is padded to its longest member, and the attention mask
    keeps padding from affecting the result (up to float rounding).
    """

    def __init__(self, model_name: str = FINBERT_MODEL, *, revision: str | None = None,
                 device: str = "cpu", cache_dir: str | None = None):
        self.model_name = model_name
        self.device = resolve_device(device)
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, revision=revision, cache_dir=cache_dir)
        except Exception as e:  # import, network, cache or format problems
            raise load_error(model_name, e) from None

        model.eval()
        self._model = model.to(self.device)
        id2label = model.config.id2label
        self.labels = tuple(str(id2label[i]).lower() for i in range(len(id2label)))
        # resolved hub commit of the weights actually loaded (falls back to the requested revision)
        self.model_revision = getattr(model.config, "_commit_hash", None) or revision
        limits = [n for n in (getattr(self._tokenizer, "model_max_length", None),
                              getattr(model.config, "max_position_embeddings", None))
                  if isinstance(n, int) and 0 < n < 100_000]
        self.max_length = min(limits) if limits else 512

    def predict_logits(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self._tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_length,
            padding="longest" if len(texts) > 1 else False,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with self._torch.inference_mode():
            logits = self._model(**encoded).logits
        return logits.detach().to("cpu", dtype=self._torch.float64).numpy()


# ==========================================
# Service
# ==========================================


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax in float64."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class FinBertSentimentService:
    """Scores texts/articles with one reusable backend. Output order == input order."""

    def __init__(self, backend: SentimentBackend, *, batch_size: int = DEFAULT_BATCH_SIZE):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        labels = tuple(str(label).lower() for label in backend.labels)
        if sorted(labels) != sorted(LABELS):
            raise SentimentModelError(f"backend labels {labels} are not exactly {LABELS}")
        self.backend = backend
        self.batch_size = batch_size
        self._columns = [labels.index(label) for label in LABELS]   # backend column of each LABEL

    @classmethod
    def load(cls, *, model_name: str = FINBERT_MODEL, revision: str | None = None, device: str = "cpu",
             batch_size: int = DEFAULT_BATCH_SIZE) -> "FinBertSentimentService":
        return cls(TransformersFinBertBackend(model_name, revision=revision, device=device), batch_size=batch_size)

    # ---------------- scoring ----------------

    def score_texts(self, texts: Iterable[str]) -> list[SentimentResult]:
        texts = list(texts)
        for i, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                raise SentimentTextError(f"text #{i} is empty")
        results: list[SentimentResult] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            results.extend(self._score_batch(batch))
        return results

    def score_text(self, text: str) -> SentimentResult:
        return self.score_texts([text])[0]

    def score_articles(self, articles: Iterable[NewsArticle]) -> list[SentimentResult]:
        """Results in the same order as `articles`; pair them with article_key externally."""
        return self.score_texts([build_sentiment_text(a) for a in articles])

    def score_article(self, article: NewsArticle) -> SentimentResult:
        return self.score_articles([article])[0]

    def _score_batch(self, batch: list[str]) -> list[SentimentResult]:
        try:
            logits = np.asarray(self.backend.predict_logits(batch), dtype=np.float64)
        except SentimentModelError:
            raise
        except Exception as e:
            raise SentimentModelError(f"inference failed ({type(e).__name__}): {redact_secrets(str(e))[:300]}") from None

        expected = (len(batch), len(LABELS))
        if logits.shape != expected:
            raise SentimentModelError(f"model returned logits of shape {logits.shape}, expected {expected}")
        if not np.isfinite(logits).all():
            raise SentimentModelError("model returned non-finite logits")

        probs = softmax(logits[:, self._columns])      # columns now in LABELS order
        out = []
        for text, row in zip(batch, probs):
            pos, neg, neu = (float(x) for x in row)
            label = max(LABELS, key=lambda lbl: (row[LABELS.index(lbl)], -LABELS.index(lbl)))
            out.append(SentimentResult(
                inference_version=INFERENCE_VERSION,
                model_name=self.backend.model_name,
                model_revision=self.backend.model_revision,
                label=label,
                positive_probability=pos,
                negative_probability=neg,
                neutral_probability=neu,
                sentiment_score=pos - neg,
                input_text_hash=text_hash(text),
            ))
        return out
