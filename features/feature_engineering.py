import math

import numpy as np

from services.market_data import (
    InsufficientHistoryError,
    MarketDataError,
    validate_ohlcv,
)


# ==========================================
# Technical feature schema
# ==========================================
# Bump TECHNICAL_FEATURE_VERSION whenever a feature is added, removed,
# renamed, or its formula changes. The evaluation harness records it
# with every result so experiments stay comparable.
#
# technical_v2 (2026-09-30):
#   - Return_Lag_k        -> Return_k         (they are cumulative k-day
#   - Volume_Change_Lag_k -> Volume_Change_k   changes, not 1-day lags)
#   - removed Daily_Return (exact duplicate of Return_1)
#   - removed Dividends / Stock Splits (raw yfinance columns, not
#     technical predictors)

TECHNICAL_FEATURE_VERSION = "technical_v2"

TECHNICAL_FEATURES = [
    "Close_to_SMA_7",
    "Close_to_SMA_30",
    "Close_to_EMA_12",
    "Close_to_EMA_26",
    "MACD_Norm",
    "MACD_Signal_Norm",
    "MACD_Histogram_Norm",
    "RSI",
    "BB_Position",
    "BB_Width",
    "Return_1",
    "Return_2",
    "Return_3",
    "Return_5",
    "Return_10",
    "Volume_Change_1",
    "Volume_Change_2",
    "Volume_Change_5",
    "Log_Return",
    "Volatility",
]

# Raw yfinance columns that must never reach the model
NON_PREDICTIVE_COLUMNS = ["Dividends", "Stock Splits", "Capital Gains"]


# ==========================================
# Minimum history (Phase 2)
# ==========================================
# Window sizes used by the formulas below. MIN_HISTORY_ROWS is DERIVED
# from them, so changing a window automatically changes the requirement.

SMA_WINDOWS = (7, 30)
EMA_SPANS = (12, 26)
MACD_SIGNAL_SPAN = 9
RSI_WINDOW = 14
BOLLINGER_WINDOW = 20
RETURN_HORIZONS = (1, 2, 3, 5, 10)
VOLUME_CHANGE_HORIZONS = (1, 2, 5)
VOLATILITY_WINDOW = 20

# An EMA with adjust=False starts at the first price; the weight of that
# arbitrary starting value after n steps is (1 - 2/(span+1))**n.
# A row is only used once that weight is below EMA_TOLERANCE.
EMA_TOLERANCE = 0.01


def ema_warmup_rows(span: int, tolerance: float = EMA_TOLERANCE) -> int:
    """Steps until an EMA's initial value carries less than `tolerance` weight."""
    return math.ceil(math.log(tolerance) / math.log(1 - 2 / (span + 1)))


# Rows needed before the first row where every rolling window is full
ROLLING_LOOKBACK_ROWS = max(
    max(SMA_WINDOWS),
    RSI_WINDOW + 1,                  # diff() then rolling(14)
    BOLLINGER_WINDOW,
    max(RETURN_HORIZONS) + 1,
    max(VOLUME_CHANGE_HORIZONS) + 1,
    VOLATILITY_WINDOW + 1,           # pct_change() then rolling(20)
)

# Slowest EMA (26) must converge, then the MACD signal EMA (9) built on it
EMA_WARMUP_ROWS = ema_warmup_rows(max(EMA_SPANS)) + ema_warmup_rows(MACD_SIGNAL_SPAN)

# Raw daily bars needed to produce ONE valid feature row (the last one).
# With n >= MIN_HISTORY_ROWS input bars, engineer_features returns
# n - MIN_HISTORY_ROWS + 1 rows.
MIN_HISTORY_ROWS = max(ROLLING_LOOKBACK_ROWS, EMA_WARMUP_ROWS)


class FeatureGenerationError(MarketDataError):
    """Features could not be computed without NaN/inf for a usable row."""


def check_min_history(n_rows: int) -> None:
    """Raise InsufficientHistoryError if `n_rows` bars cannot produce a feature row."""
    if n_rows < MIN_HISTORY_ROWS:
        raise InsufficientHistoryError(
            f"Need at least {MIN_HISTORY_ROWS} completed daily bars to compute "
            f"technical features ({TECHNICAL_FEATURE_VERSION}); got {n_rows}. "
            "Fetch a longer period (e.g. period='1y')."
        )


def engineer_features(df):
    """
    Adds technical indicators to the stock dataframe.

    IMPORTANT (fix applied):
    Most indicators are now expressed as RATIOS / DIFFERENCES RELATIVE
    TO PRICE (or otherwise normalized) rather than as raw price levels.

    Tree-based models (Random Forest, XGBoost, LightGBM) predict by
    averaging target values seen in training leaves - they cannot
    extrapolate beyond the numeric range seen during training. Raw
    price-level features (e.g. SMA_7 = 185.32) keep growing as a stock
    trends upward over years, so a tree trained on older/lower price
    levels has no way to represent new, higher price levels in a later
    test or live period. It just "caps out" near the highest value it
    saw in training. Normalizing removes this ceiling.

    Phase 2 contract:
        - input must satisfy the market-data contract
          (services/market_data.py: canonical columns, sorted, unique
          dates, no NaN/inf, positive prices/volume)
        - at least MIN_HISTORY_ROWS bars are required, otherwise
          InsufficientHistoryError
        - the first MIN_HISTORY_ROWS - 1 rows (rolling windows not full /
          EMAs not converged) are removed; every returned row has finite
          values for all TECHNICAL_FEATURES, otherwise
          FeatureGenerationError. Rows are never dropped silently.
        - each row uses only that day and earlier days (no look-ahead)

    Parameters:
        df (pd.DataFrame): Daily bars (Date, Open, High, Low, Close, Volume;
            extra raw columns such as Dividends are ignored)

    Returns:
        pd.DataFrame: Date, OHLCV and TECHNICAL_FEATURES, one row per bar
            from bar number MIN_HISTORY_ROWS onwards
    """

    df = df.drop(columns=[c for c in NON_PREDICTIVE_COLUMNS if c in df.columns])
    validate_ohlcv(df)
    check_min_history(len(df))
    df = df.reset_index(drop=True)

    close = df["Close"]

    # =========================
    # Moving Averages -> normalized as ratio to Close
    # =========================

    sma_7 = close.rolling(window=SMA_WINDOWS[0]).mean()
    sma_30 = close.rolling(window=SMA_WINDOWS[1]).mean()
    ema_12 = close.ewm(span=EMA_SPANS[0], adjust=False).mean()
    ema_26 = close.ewm(span=EMA_SPANS[1], adjust=False).mean()

    df["Close_to_SMA_7"] = close / sma_7
    df["Close_to_SMA_30"] = close / sma_30
    df["Close_to_EMA_12"] = close / ema_12
    df["Close_to_EMA_26"] = close / ema_26

    # =========================
    # MACD -> normalized by Close so it's scale-free
    # =========================

    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=MACD_SIGNAL_SPAN, adjust=False).mean()
    macd_hist = macd - macd_signal

    df["MACD_Norm"] = macd / close
    df["MACD_Signal_Norm"] = macd_signal / close
    df["MACD_Histogram_Norm"] = macd_hist / close

    # =========================
    # RSI -> already bounded 0-100, no change needed
    # =========================

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=RSI_WINDOW).mean()
    avg_loss = loss.rolling(window=RSI_WINDOW).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # =========================
    # Bollinger Bands -> position within band + relative width
    # =========================

    rolling_mean = close.rolling(window=BOLLINGER_WINDOW).mean()
    rolling_std = close.rolling(window=BOLLINGER_WINDOW).std()

    bb_upper = rolling_mean + (2 * rolling_std)
    bb_lower = rolling_mean - (2 * rolling_std)

    # Where Close currently sits within the band: 0 = lower band, 1 = upper band
    df["BB_Position"] = (close - bb_lower) / (bb_upper - bb_lower)

    # Band width relative to price (volatility proxy, scale-free)
    df["BB_Width"] = (bb_upper - bb_lower) / close

    # =========================
    # k-day returns (cumulative change over the last k days)
    # =========================

    for k in RETURN_HORIZONS:
        df[f"Return_{k}"] = close.pct_change(k)

    # Volume as relative change over k days instead of raw share counts
    for k in VOLUME_CHANGE_HORIZONS:
        df[f"Volume_Change_{k}"] = df["Volume"].pct_change(k)

    df["Log_Return"] = np.log(close / close.shift(1))

    # =========================
    # Volatility (20-day std of 1-day returns)
    # =========================

    df["Volatility"] = (
        df["Return_1"]
        .rolling(window=VOLATILITY_WINDOW)
        .std()
    )

    # =========================
    # Drop warm-up rows, then require every remaining row to be finite
    # =========================

    df = df.iloc[MIN_HISTORY_ROWS - 1:].reset_index(drop=True)

    values = df[TECHNICAL_FEATURES].to_numpy(dtype=float)
    bad_rows = ~np.isfinite(values).all(axis=1)
    if bad_rows.any():
        bad_features = [f for f, bad in zip(TECHNICAL_FEATURES, ~np.isfinite(values).all(axis=0)) if bad]
        bad_dates = df.loc[bad_rows, "Date"].astype(str).head(5).tolist()
        raise FeatureGenerationError(
            f"{int(bad_rows.sum())} row(s) have NaN/inf features {bad_features} "
            f"(first dates: {bad_dates})"
        )

    return df