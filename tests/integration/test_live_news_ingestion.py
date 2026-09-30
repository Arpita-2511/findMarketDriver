"""
LIVE Phase 4C run: chunked Alpaca ingestion -> manifest -> canonical dataset,
written to a pytest temp directory (never to data/). Needs internet and
ALPACA_API_KEY / ALPACA_API_SECRET in .env. Opt-in:

    pytest -m integration tests/integration/test_live_news_ingestion.py
"""

import os
from datetime import datetime, timezone

import pytest

import config.settings  # noqa: F401  (loads .env)
from services.alpaca_news_provider import API_KEY_ENV, API_SECRET_ENV, AlpacaNewsProvider
from services.historical_news_ingestion import IngestionConfig, run_historical_ingestion
from services.news_dataset import build_canonical_dataset, load_canonical_dataset

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (os.getenv(API_KEY_ENV) and os.getenv(API_SECRET_ENV)),
                       reason="Alpaca credentials not configured"),
]

UTC = timezone.utc


def test_live_chunked_ingestion_builds_canonical_dataset(tmp_path):
    cfg = IngestionConfig(provider="alpaca", symbols=("AAPL",), start=datetime(2017, 1, 3, tzinfo=UTC),
                          end=datetime(2017, 1, 10, tzinfo=UTC), chunk_days=3, page_size=20)
    manifest = run_historical_ingestion(AlpacaNewsProvider.from_env(), cfg, raw_dir=tmp_path / "raw")
    path, meta = build_canonical_dataset(manifest, tmp_path / "processed")
    articles, _ = load_canonical_dataset(path)

    assert meta["rows"] == len(articles) > 0
    assert len({a.key for a in articles}) == len(articles)
    assert all(cfg.start <= a.created_at < cfg.end and "AAPL" in a.symbols for a in articles)
    assert [a.created_at for a in articles] == sorted(a.created_at for a in articles)
    print(f"chunks={len(meta['source_snapshots'])} input={meta['n_input_records']} rows={meta['rows']} "
          f"dups={meta['n_duplicates_removed']} outside={meta['n_outside_interval']}")
