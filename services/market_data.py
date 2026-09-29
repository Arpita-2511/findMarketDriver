"""
Canonical daily market-data contract (Phase 2).

Every source of daily bars - yfinance Ticker.history(), the yf.download()
fallback, a raw snapshot on disk - is converted to ONE schema here before
anything downstream (feature engineering, dataset building, prediction)
sees it:

    Date    datetime64[ns], timezone-naive, midnight = exchange-local
            (America/New_York) trading date
    Open, High, Low, Close   float64, split- and dividend-adjusted
    Volume                   float64

This module has no network dependency so it can be unit-tested with
fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]

# Timezone in which the "Date" of a daily bar is defined
EXCHANGE_TIMEZONE = "America/New_York"


class MarketDataError(ValueError):
    """Market data is missing, malformed, or violates the contract."""


class InsufficientHistoryError(MarketDataError):
    """Not enough daily bars to compute every feature reliably."""


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a yfinance-style frame to the canonical schema.

    Accepts either a DatetimeIndex (as returned by yfinance) or a 'Date'
    column, and MultiIndex columns (yf.download) or flat columns
    (Ticker.history). Extra columns such as Dividends / Stock Splits are
    dropped. Does NOT sort or de-duplicate - validate_ohlcv() rejects
    such data instead, so problems are never silently repaired.
    """
    if raw is None or raw.empty:
        raise MarketDataError("No market data returned")

    df = raw.copy()

    if isinstance(df.columns, pd.MultiIndex):
        # yf.download: (Price, Ticker) -> keep the Price level
        df.columns = df.columns.get_level_values(0)
    df.columns.name = None

    if "Date" not in df.columns:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise MarketDataError("Market data has neither a 'Date' column nor a DatetimeIndex")
        df = df.rename_axis("Date").reset_index()

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise MarketDataError(f"Market data is missing columns: {missing}")

    df = df[OHLCV_COLUMNS].copy()
    df["Date"] = _to_exchange_date(df["Date"])

    for col in OHLCV_COLUMNS[1:]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise MarketDataError(f"Column '{col}' is not numeric")
        df[col] = df[col].astype("float64")

    return df.reset_index(drop=True)


def _to_exchange_date(dates: pd.Series) -> pd.Series:
    """Timezone-aware -> exchange-local calendar date; naive -> assumed exchange-local."""
    dates = pd.to_datetime(dates)
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert(EXCHANGE_TIMEZONE).dt.tz_localize(None)
    return dates.dt.normalize().astype("datetime64[ns]")


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce the market-data contract on canonical-schema data.
    Raises MarketDataError listing every problem; returns df unchanged.

    Deliberately NOT checked: High >= Open/Close >= Low. Adjusted
    yfinance data can violate this by rounding, so it would reject
    legitimate history.
    """
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise MarketDataError(f"Market data is missing columns: {missing}")
    if df.empty:
        raise MarketDataError("Market data is empty")

    problems: list[str] = []

    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        raise MarketDataError("'Date' must be a datetime column")

    n_dup = int(df["Date"].duplicated().sum())
    if n_dup:
        problems.append(f"{n_dup} duplicate date(s)")
    if not df["Date"].is_monotonic_increasing:
        problems.append("dates are not in chronological order")

    values = df[OHLCV_COLUMNS[1:]]
    non_numeric = [c for c in values.columns if not pd.api.types.is_numeric_dtype(values[c])]
    if non_numeric:
        raise MarketDataError(f"Non-numeric columns: {non_numeric}")

    nan_cols = values.columns[values.isna().any()].tolist()
    if nan_cols:
        problems.append(f"NaN values in {nan_cols}")
    inf_cols = values.columns[np.isinf(values).any()].tolist()
    if inf_cols:
        problems.append(f"infinite values in {inf_cols}")

    if (df[PRICE_COLUMNS] <= 0).any().any():
        problems.append("non-positive prices")
    # Zero volume would make Volume_Change infinite
    if (df["Volume"] <= 0).any():
        problems.append("non-positive volume")

    if problems:
        raise MarketDataError("Invalid market data: " + "; ".join(problems))
    return df
