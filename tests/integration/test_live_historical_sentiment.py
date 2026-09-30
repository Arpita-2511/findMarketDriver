"""
REAL ProsusAI/finbert through the Phase 5C historical pipeline, on a tiny
local fixture (3 articles) written to a pytest temp directory. No API keys.
Uses the cached model (downloads once if not cached). Opt-in:

    pytest -m integration tests/integration/test_live_historical_sentiment.py -s
"""

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from services.finbert_sentiment import LABELS, FinBertSentimentService
from services.historical_sentiment import load_sentiment_dataset, run_historical_sentiment
from tests.conftest import calendar_2024
from tests.sentiment_fakes import bars_for, news, write_canonical

pytestmark = pytest.mark.integration

NY = ZoneInfo("America/New_York")


@pytest.fixture(scope="module")
def finbert():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return FinBertSentimentService.load(device="cpu", batch_size=2)


def test_real_model_through_historical_pipeline(finbert, tmp_path):
    articles = [
        news(1, datetime(2024, 7, 9, 10, 0, tzinfo=NY), "Apple reports record revenue and beats analyst expectations"),
        news(2, datetime(2024, 7, 10, 11, 0, tzinfo=NY), "Apple shares plunge after the company cuts its outlook"),
        news(3, datetime(2024, 7, 10, 17, 30, tzinfo=NY), "Apple will hold its annual shareholder meeting"),
    ]
    canonical = write_canonical(tmp_path / "news", articles, start=datetime(2024, 7, 8, tzinfo=timezone.utc),
                                end=datetime(2024, 7, 13, tzinfo=timezone.utc))
    sessions = [d for d in pd.bdate_range("2024-07-01", "2024-07-31").date if calendar_2024().is_session(d)]
    out = run_historical_sentiment(canonical, bars_for(sessions), finbert, output_dir=tmp_path / "out",
                                   calendar_source={"file": "fixture", "sha256": "0" * 64})

    meta = out["sentiment_meta"]
    assert meta["rows"] == 3 and meta["provenance"]["model_name"] == "ProsusAI/finbert"
    assert meta["provenance"]["model_revision"]

    scored, _ = load_sentiment_dataset(out["sentiment_path"], canonical)
    for s in scored:
        assert s.sentiment.label in LABELS and -1.0 <= s.sentiment.sentiment_score <= 1.0
    assert scored[0].sentiment.sentiment_score > scored[1].sentiment.sentiment_score   # positive vs negative text

    rows = {r["trading_date"]: r for r in map(json.loads, out["daily_path"].read_text(encoding="utf-8").splitlines())}
    assert rows["2024-07-10"]["news_count"] == 2           # 17:30 article not yet eligible
    assert rows["2024-07-11"]["news_count"] == 3
    print(f"\nrevision={meta['provenance']['model_revision']}  labels="
          f"{[s.sentiment.label for s in scored]}  daily rows={out['daily_meta']['rows']}")
