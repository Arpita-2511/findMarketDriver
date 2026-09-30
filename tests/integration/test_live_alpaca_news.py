"""
LIVE Alpaca news test - needs internet and ALPACA_API_KEY / ALPACA_API_SECRET
in .env. Excluded by default; run with:

    pytest -m integration tests/integration/test_live_alpaca_news.py

Skipped (not failed) when credentials are absent. Never prints credentials.
"""

import os
from datetime import datetime, timezone

import pytest

import config.settings  # noqa: F401  (loads .env)
from services.alpaca_news_provider import API_KEY_ENV, API_SECRET_ENV, AlpacaNewsProvider
from services.news_provider import NewsQuery

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (os.getenv(API_KEY_ENV) and os.getenv(API_SECRET_ENV)),
                       reason="Alpaca credentials not configured"),
]

UTC = timezone.utc


def test_live_historical_page_walk_returns_canonical_articles():
    query = NewsQuery(symbols=("AAPL",), start=datetime(2017, 1, 3, tzinfo=UTC),
                      end=datetime(2017, 1, 10, tzinfo=UTC), page_size=10)
    result = AlpacaNewsProvider.from_env().fetch_historical(query, max_pages=50)
    p = result.provenance

    assert p["n_articles"] > 0
    assert p["n_created_outside_window"] == 0
    assert len({a.key for a in result.articles}) == len(result.articles)
    for a in result.articles:
        assert "AAPL" in a.symbols
        assert a.created_at.tzinfo == UTC and a.information_available_at >= a.created_at
        assert a.headline
    print(f"pages={p['pages']} articles={p['n_articles']} raw={p['n_raw_articles']} "
          f"duplicates={p['n_duplicates_removed']}")
