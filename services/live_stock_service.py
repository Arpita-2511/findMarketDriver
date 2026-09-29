import time
from datetime import datetime

import yfinance as yf

from services.market_calendar_service import drop_incomplete_bars
from services.market_data import MarketDataError, normalize_ohlcv, validate_ohlcv

# ==========================================
# Price semantics (Phase 2)
# ==========================================
# Both fetch paths request auto_adjust=True, so Open/High/Low/Close are
# split- AND dividend-adjusted as of the moment of retrieval. Returns
# computed from them are total returns (dividends included).
#
# Adjusted history is back-adjusted: when a new dividend or split
# occurs, Yahoo rescales ALL earlier prices. Consequences:
#   - Ratios/returns computed WITHIN one retrieval are consistent.
#   - Prices from DIFFERENT retrievals must not be mixed (e.g. a close
#     logged last month vs. a close fetched today). Compute returns from
#     one series only - see models/evaluate_live_tracking.py.
#   - Training reproducibility therefore relies on a stored raw snapshot
#     (training/build_dataset.py), not on re-downloading.

YF_PARAMS = {"interval": "1d", "auto_adjust": True}


def _fetch_history(symbol, period):
    df = yf.Ticker(symbol).history(period=period, **YF_PARAMS)
    return df


def _fetch_download(symbol, period):
    return yf.download(symbol, period=period, progress=False, **YF_PARAMS)


def fetch_latest_stock_data(symbol, period="1y", *, now: datetime | None = None,
                            completed_only=True, retries=3, retry_delay=2.0):
    """
    Fetch daily bars from Yahoo Finance in the canonical schema
    (services/market_data.py): Date, Open, High, Low, Close, Volume.

    Parameters
    ----------
    symbol : str
        Stock ticker
    period : str
        yfinance period, e.g. 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y
    now : datetime, optional
        Timezone-aware "current time" for the completed-bar rule
        (defaults to the real clock; tests inject a fixed time).
    completed_only : bool
        Drop the current session's bar if it is not complete yet
        (services/market_calendar_service.py). Default True.

    Returns
    -------
    pandas.DataFrame with attrs["source_method"] set to
    "Ticker.history" or "yf.download".

    Raises
    ------
    MarketDataError if neither path returns valid data.
    """
    errors = []

    # ==========================================
    # First Attempt : Ticker.history()
    # ==========================================

    for attempt in range(retries):
        try:
            raw = _fetch_history(symbol, period)
        except Exception as e:  # network / yfinance errors: record and retry
            errors.append(f"Ticker.history() attempt {attempt + 1}: {e}")
            raw = None

        if raw is not None and not raw.empty:
            return _finalize(raw, "Ticker.history", now, completed_only)

        if raw is not None:
            errors.append(f"Ticker.history() attempt {attempt + 1}: empty result")
        print(f"Retry {attempt + 1}/{retries}...")
        time.sleep(retry_delay)

    # ==========================================
    # Second Attempt : yf.download()
    # ==========================================

    try:
        raw = _fetch_download(symbol, period)
    except Exception as e:
        errors.append(f"yf.download(): {e}")
        raw = None

    if raw is not None and not raw.empty:
        print("⚠ Using yf.download() fallback")
        return _finalize(raw, "yf.download", now, completed_only)

    if raw is not None:
        errors.append("yf.download(): empty result")

    # ==========================================
    # Final Failure
    # ==========================================

    raise MarketDataError(
        f"Unable to fetch stock data for '{symbol}'. " + " | ".join(errors)
    )


def _finalize(raw, source_method, now, completed_only):
    """Same normalization + validation for every source path."""
    df = validate_ohlcv(normalize_ohlcv(raw))
    df.attrs["source_method"] = source_method
    if completed_only:
        df = drop_incomplete_bars(df, now)
    return df


if __name__ == "__main__":

    symbol = input("Enter Stock Symbol : ").upper()

    stock_df = fetch_latest_stock_data(symbol)

    print(stock_df.tail())

    print("\nColumns:")
    print(stock_df.columns.tolist())
    print("Source:", stock_df.attrs.get("source_method"),
          "| incomplete bars dropped:", stock_df.attrs.get("incomplete_bars_dropped"))
