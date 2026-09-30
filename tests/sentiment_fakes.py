"""Offline fixtures for Phase 5C: fake FinBERT backend, canonical-dataset writer, market bars."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from services.finbert_sentiment import FinBertSentimentService
from services.news_dataset import serialize_dataset, sort_key
from services.news_schema import NewsArticle, historical_information_available_at

FETCHED = datetime(2026, 10, 1, tzinfo=timezone.utc)


class FakeFinBert:
    """Deterministic keyword logits (unique per text); same provenance shape as the real backend."""

    model_name = "ProsusAI/finbert"

    def __init__(self, revision="fakerev000001", labels=("positive", "negative", "neutral"), bias=0.0):
        self.model_revision = revision
        self.labels = labels
        self.bias = bias
        self.calls: list[list[str]] = []

    def predict_logits(self, texts):
        self.calls.append(list(texts))
        rows = []
        for t in texts:
            low = t.lower()
            s = {"positive": 3.0 if "beats" in low else 0.0,
                 "negative": 3.0 if "plunge" in low else 0.0,
                 "neutral": 1.0 + (len(t) % 7) / 10 + self.bias}
            rows.append([s[label] for label in self.labels])
        return np.array(rows)


def fake_service(batch_size=16, **backend_kwargs) -> FinBertSentimentService:
    return FinBertSentimentService(FakeFinBert(**backend_kwargs), batch_size=batch_size)


def news(i, when, headline=None, symbols=("AAPL",), updated=None, fetched=FETCHED, summary=None) -> NewsArticle:
    return NewsArticle(provider="alpaca", provider_article_id=str(i), headline=headline or f"Apple update {i}",
                       summary=summary, symbols=symbols, created_at=when, updated_at=updated,
                       information_available_at=historical_information_available_at(when, updated),
                       fetched_at=fetched)


def write_canonical(directory: Path, articles, *, start: datetime, end: datetime, symbols=("AAPL",),
                    name="alpaca_AAPL_test.jsonl") -> Path:
    """Minimal Phase 4C-style canonical dataset (+ .meta.json) that load_canonical_dataset accepts."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = serialize_dataset(sorted(articles, key=sort_key))
    path = directory / name
    path.write_bytes(payload)
    meta = {"run_id": name.removesuffix(".jsonl"), "symbols": list(symbols), "start": start.isoformat(),
            "end": end.isoformat(), "rows": len(articles), "sha256": hashlib.sha256(payload).hexdigest()}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return path


def bars_for(dates) -> pd.DataFrame:
    """Valid canonical OHLCV bars whose dates define the trading sessions."""
    n = len(dates)
    close = np.linspace(100.0, 110.0, n)
    return pd.DataFrame({"Date": pd.to_datetime(list(dates)).astype("datetime64[ns]"), "Open": close,
                         "High": close * 1.01, "Low": close * 0.99, "Close": close,
                         "Volume": np.full(n, 1_000_000.0)})
