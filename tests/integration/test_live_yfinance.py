"""
LIVE integration tests - need internet access, no API key.
Excluded by default; run with:  pytest -m integration
"""

from datetime import datetime, timezone

import pytest

import services.live_stock_service as live
from features.feature_engineering import MIN_HISTORY_ROWS, TECHNICAL_FEATURES, engineer_features
from services.market_calendar_service import is_bar_complete
from services.market_data import OHLCV_COLUMNS, normalize_ohlcv, validate_ohlcv

pytestmark = pytest.mark.integration


def test_live_fetch_returns_completed_canonical_bars():
    now = datetime.now(timezone.utc)
    df = live.fetch_latest_stock_data("AAPL", period="1y", now=now)
    assert list(df.columns) == OHLCV_COLUMNS
    assert len(df) >= MIN_HISTORY_ROWS
    assert is_bar_complete(df["Date"].iloc[-1], now)
    features = engineer_features(df)
    assert set(TECHNICAL_FEATURES) <= set(features.columns)


def test_live_primary_and_fallback_share_schema():
    primary = validate_ohlcv(normalize_ohlcv(live._fetch_history("AAPL", "3mo")))
    fallback = validate_ohlcv(normalize_ohlcv(live._fetch_download("AAPL", "3mo")))
    assert list(primary.columns) == list(fallback.columns) == OHLCV_COLUMNS
    assert (primary.dtypes == fallback.dtypes).all()
    shared = primary.merge(fallback, on="Date", suffixes=("_h", "_d"))
    assert len(shared) > 40
