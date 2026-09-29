"""Market-data contract, source normalization and fallback behaviour (no network)."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import services.live_stock_service as live
from services.market_data import (
    OHLCV_COLUMNS,
    MarketDataError,
    normalize_ohlcv,
    validate_ohlcv,
)
from tests.conftest import as_ticker_history, as_yf_download, make_bars

AFTER_DATA = datetime(2025, 1, 1, tzinfo=timezone.utc)  # every fixture bar is complete


# ==========================================
# Normalization: both yfinance shapes -> one schema
# ==========================================


def test_history_shape_normalizes_to_canonical_schema(bars):
    out = normalize_ohlcv(as_ticker_history(bars))
    assert list(out.columns) == OHLCV_COLUMNS
    assert out["Date"].dt.tz is None
    assert all(out[c].dtype == "float64" for c in OHLCV_COLUMNS[1:])
    pd.testing.assert_frame_equal(out, bars)


def test_download_shape_normalizes_to_identical_frame(bars):
    from_history = normalize_ohlcv(as_ticker_history(bars))
    from_download = normalize_ohlcv(as_yf_download(bars))
    pd.testing.assert_frame_equal(from_history, from_download)


def test_timezone_aware_dates_keep_exchange_date():
    # 00:00 New York == 04:00/05:00 UTC; converting to UTC first must not shift the date
    bars = make_bars(5)
    raw = as_ticker_history(bars)
    raw.index = raw.index.tz_convert("UTC")
    assert (normalize_ohlcv(raw)["Date"] == bars["Date"]).all()


def test_normalize_rejects_missing_columns(bars):
    with pytest.raises(MarketDataError, match="missing columns"):
        normalize_ohlcv(bars.drop(columns="Volume"))


def test_normalize_rejects_empty():
    with pytest.raises(MarketDataError, match="No market data"):
        normalize_ohlcv(pd.DataFrame())


def test_normalize_rejects_non_numeric(bars):
    with pytest.raises(MarketDataError, match="not numeric"):
        normalize_ohlcv(bars.assign(Close="n/a"))


# ==========================================
# Validation
# ==========================================


def test_valid_bars_pass(bars):
    assert validate_ohlcv(bars) is bars


@pytest.mark.parametrize("corrupt, message", [
    (lambda d: d.assign(Close=d["Close"].where(d.index != 3, np.nan)), "NaN"),
    (lambda d: d.assign(High=d["High"].where(d.index != 4, np.inf)), "infinite"),
    (lambda d: d.assign(Date=d["Date"].where(d.index != 6, d["Date"].iloc[5])), "duplicate"),
    (lambda d: d.iloc[::-1].reset_index(drop=True), "chronological"),
    (lambda d: d.assign(Low=d["Low"].where(d.index != 2, -1.0)), "non-positive prices"),
    (lambda d: d.assign(Volume=d["Volume"].where(d.index != 2, 0.0)), "non-positive volume"),
])
def test_invalid_bars_rejected(bars, corrupt, message):
    with pytest.raises(MarketDataError, match=message):
        validate_ohlcv(corrupt(bars))


def test_validation_rejects_string_dates(bars):
    with pytest.raises(MarketDataError, match="datetime"):
        validate_ohlcv(bars.assign(Date=bars["Date"].astype(str)))


# ==========================================
# fetch_latest_stock_data: primary path, fallback, failure
# ==========================================


def _patch_sources(monkeypatch, history, download):
    monkeypatch.setattr(live, "_fetch_history", history)
    monkeypatch.setattr(live, "_fetch_download", download)


def _raise(*_):
    raise ConnectionError("network down")


def test_primary_path(monkeypatch, bars):
    _patch_sources(monkeypatch, lambda s, p: as_ticker_history(bars), _raise)
    out = live.fetch_latest_stock_data("AAPL", now=AFTER_DATA, retry_delay=0)
    assert out.attrs["source_method"] == "Ticker.history"
    pd.testing.assert_frame_equal(out, bars)


@pytest.mark.parametrize("primary", [_raise, lambda s, p: pd.DataFrame()])
def test_fallback_schema_equals_primary_schema(monkeypatch, bars, primary):
    _patch_sources(monkeypatch, lambda s, p: as_ticker_history(bars), _raise)
    normal = live.fetch_latest_stock_data("AAPL", now=AFTER_DATA, retry_delay=0)

    _patch_sources(monkeypatch, primary, lambda s, p: as_yf_download(bars))
    fallback = live.fetch_latest_stock_data("AAPL", now=AFTER_DATA, retry_delay=0)

    assert fallback.attrs["source_method"] == "yf.download"
    assert list(fallback.columns) == list(normal.columns)
    assert (fallback.dtypes == normal.dtypes).all()
    pd.testing.assert_frame_equal(fallback, normal)


def test_both_paths_fail_raises_market_data_error(monkeypatch):
    _patch_sources(monkeypatch, _raise, lambda s, p: pd.DataFrame())
    with pytest.raises(MarketDataError, match="Unable to fetch stock data for 'ZZZZ'"):
        live.fetch_latest_stock_data("ZZZZ", now=AFTER_DATA, retry_delay=0)


def test_malformed_source_data_is_rejected_not_repaired(monkeypatch, bars):
    shuffled = as_ticker_history(bars).iloc[::-1]
    _patch_sources(monkeypatch, lambda s, p: shuffled, _raise)
    with pytest.raises(MarketDataError, match="chronological"):
        live.fetch_latest_stock_data("AAPL", now=AFTER_DATA, retry_delay=0)
