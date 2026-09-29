"""Minimum-history policy and canonical technical_v2 schema."""

import math

import numpy as np
import pandas as pd
import pytest

from features.feature_engineering import (
    EMA_WARMUP_ROWS,
    MIN_HISTORY_ROWS,
    ROLLING_LOOKBACK_ROWS,
    TECHNICAL_FEATURES,
    FeatureGenerationError,
    ema_warmup_rows,
    engineer_features,
)
from services.market_data import OHLCV_COLUMNS, InsufficientHistoryError, MarketDataError
from services.market_data import normalize_ohlcv
from tests.conftest import as_ticker_history, as_yf_download, make_bars


# ==========================================
# Derivation of the requirement
# ==========================================


def test_min_history_is_derived_from_the_pipeline():
    assert ROLLING_LOOKBACK_ROWS == 30                    # SMA_30 is the longest window
    assert ema_warmup_rows(26) == 60 and ema_warmup_rows(9) == 21
    assert EMA_WARMUP_ROWS == 81
    assert MIN_HISTORY_ROWS == max(ROLLING_LOOKBACK_ROWS, EMA_WARMUP_ROWS) == 81


def test_ema_warmup_meets_tolerance():
    for span in (9, 12, 26):
        n = ema_warmup_rows(span)
        decay = 1 - 2 / (span + 1)
        assert decay ** n < 0.01 <= decay ** (n - 1)
    assert math.isclose(ema_warmup_rows(26, 0.5), math.ceil(math.log(0.5) / math.log(25 / 27)))


# ==========================================
# Sufficient / insufficient / boundary
# ==========================================


def test_sufficient_history():
    out = engineer_features(make_bars(200))
    assert len(out) == 200 - MIN_HISTORY_ROWS + 1


def test_boundary_exactly_min_history_gives_one_row():
    bars = make_bars(MIN_HISTORY_ROWS)
    out = engineer_features(bars)
    assert len(out) == 1
    assert out["Date"].iloc[0] == bars["Date"].iloc[-1]


def test_one_bar_short_is_rejected_with_clear_message():
    with pytest.raises(InsufficientHistoryError, match=f"at least {MIN_HISTORY_ROWS}.*got {MIN_HISTORY_ROWS - 1}"):
        engineer_features(make_bars(MIN_HISTORY_ROWS - 1))


def test_first_returned_row_is_bar_number_min_history():
    bars = make_bars(150)
    out = engineer_features(bars)
    assert out["Date"].iloc[0] == bars["Date"].iloc[MIN_HISTORY_ROWS - 1]


# ==========================================
# No silent row dropping / malformed input
# ==========================================


def test_nan_features_raise_instead_of_silently_dropping_rows():
    bars = make_bars(150)
    flat = bars.index >= 120            # 30 identical closes -> zero Bollinger width
    bars.loc[flat, "Close"] = 100.0
    with pytest.raises(FeatureGenerationError, match="NaN/inf"):
        engineer_features(bars)


@pytest.mark.parametrize("corrupt", [
    lambda d: d.iloc[::-1].reset_index(drop=True),
    lambda d: d.assign(Close=d["Close"].where(d.index != 50, np.nan)),
    lambda d: d.assign(Date=d["Date"].where(d.index != 60, d["Date"].iloc[59])),
])
def test_malformed_input_rejected(corrupt):
    with pytest.raises(MarketDataError):
        engineer_features(corrupt(make_bars(150)))


# ==========================================
# Canonical schema across paths
# ==========================================


def test_output_schema_is_canonical():
    bars = make_bars(150).assign(Dividends=0.0, **{"Stock Splits": 0.0})
    out = engineer_features(bars)
    assert list(out.columns) == OHLCV_COLUMNS + TECHNICAL_FEATURES
    assert np.isfinite(out[TECHNICAL_FEATURES].to_numpy()).all()


def test_primary_and_fallback_paths_give_identical_features():
    bars = make_bars(150)
    via_history = engineer_features(normalize_ohlcv(as_ticker_history(bars)))
    via_download = engineer_features(normalize_ohlcv(as_yf_download(bars)))
    pd.testing.assert_frame_equal(via_history, via_download)


def test_live_window_matches_historical_features_on_same_dates():
    """
    A shorter live window (e.g. 1y) and the long historical build agree
    on shared dates to within the EMA tolerance - the reason for the
    warm-up rule. Rolling-window features match exactly.
    """
    long = make_bars(400)
    hist = engineer_features(long).set_index("Date")
    live = engineer_features(long.iloc[200:].reset_index(drop=True)).set_index("Date")
    shared = live.index

    exact = [f for f in TECHNICAL_FEATURES if not f.startswith(("Close_to_EMA", "MACD"))]
    pd.testing.assert_frame_equal(live.loc[shared, exact], hist.loc[shared, exact])

    ema_based = [f for f in TECHNICAL_FEATURES if f.startswith(("Close_to_EMA", "MACD"))]
    diff = (live.loc[shared, ema_based] - hist.loc[shared, ema_based]).abs().max()
    assert (diff < 0.01).all(), diff
