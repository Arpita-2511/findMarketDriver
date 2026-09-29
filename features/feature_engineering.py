import numpy as np
import pandas as pd


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

    Parameters:
        df (pd.DataFrame): Raw stock dataframe (must contain Close, Volume)

    Returns:
        pd.DataFrame: Dataframe with engineered features
    """

    df = df.drop(columns=[c for c in NON_PREDICTIVE_COLUMNS if c in df.columns])

    close = df["Close"]

    # =========================
    # Moving Averages -> normalized as ratio to Close
    # =========================

    sma_7 = close.rolling(window=7).mean()
    sma_30 = close.rolling(window=30).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()

    df["Close_to_SMA_7"] = close / sma_7
    df["Close_to_SMA_30"] = close / sma_30
    df["Close_to_EMA_12"] = close / ema_12
    df["Close_to_EMA_26"] = close / ema_26

    # =========================
    # MACD -> normalized by Close so it's scale-free
    # =========================

    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
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

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # =========================
    # Bollinger Bands -> position within band + relative width
    # =========================

    rolling_mean = close.rolling(window=20).mean()
    rolling_std = close.rolling(window=20).std()

    bb_upper = rolling_mean + (2 * rolling_std)
    bb_lower = rolling_mean - (2 * rolling_std)

    # Where Close currently sits within the band: 0 = lower band, 1 = upper band
    df["BB_Position"] = (close - bb_lower) / (bb_upper - bb_lower)

    # Band width relative to price (volatility proxy, scale-free)
    df["BB_Width"] = (bb_upper - bb_lower) / close

    # =========================
    # k-day returns (cumulative change over the last k days)
    # =========================

    df["Return_1"] = close.pct_change(1)
    df["Return_2"] = close.pct_change(2)
    df["Return_3"] = close.pct_change(3)
    df["Return_5"] = close.pct_change(5)
    df["Return_10"] = close.pct_change(10)

    # Volume as relative change over k days instead of raw share counts
    df["Volume_Change_1"] = df["Volume"].pct_change(1)
    df["Volume_Change_2"] = df["Volume"].pct_change(2)
    df["Volume_Change_5"] = df["Volume"].pct_change(5)

    df["Log_Return"] = np.log(close / close.shift(1))

    # =========================
    # Volatility (20-day std of 1-day returns)
    # =========================

    df["Volatility"] = (
        df["Return_1"]
        .rolling(window=20)
        .std()
    )

    # =========================
    # Remove Missing / Infinite Values
    # =========================

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df