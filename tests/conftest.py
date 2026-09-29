"""Shared, deterministic market-data fixtures (no network)."""

import numpy as np
import pandas as pd
import pytest


def make_bars(n: int = 200, start: str = "2024-01-02", seed: int = 7) -> pd.DataFrame:
    """Canonical-schema daily bars (naive exchange dates, float64)."""
    rng = np.random.default_rng(seed)
    close = 150 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    return pd.DataFrame({
        "Date": pd.bdate_range(start, periods=n).astype("datetime64[ns]"),
        "Open": close * (1 + rng.normal(0, 0.003, n)),
        "High": close * 1.012,
        "Low": close * 0.988,
        "Close": close,
        "Volume": rng.integers(20_000_000, 90_000_000, n).astype("float64"),
    })


def as_ticker_history(bars: pd.DataFrame) -> pd.DataFrame:
    """Shape of yf.Ticker(...).history(): tz-aware DatetimeIndex + action columns."""
    idx = pd.DatetimeIndex(bars["Date"]).tz_localize("America/New_York")
    df = bars.drop(columns="Date").set_index(idx.rename("Date"))
    df["Volume"] = df["Volume"].astype("int64")
    df["Dividends"] = 0.0
    df["Stock Splits"] = 0.0
    return df


def as_yf_download(bars: pd.DataFrame, ticker: str = "AAPL") -> pd.DataFrame:
    """Shape of yf.download(): naive DatetimeIndex, (Price, Ticker) MultiIndex columns, other order."""
    df = bars.set_index(pd.DatetimeIndex(bars["Date"]).rename("Date"))[
        ["Close", "High", "Low", "Open", "Volume"]
    ].copy()
    df["Volume"] = df["Volume"].astype("int64")
    df.columns = pd.MultiIndex.from_product([df.columns, [ticker]], names=["Price", "Ticker"])
    return df


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_bars()
