import numpy as np
import pandas as pd


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
    # Lag Features -> expressed as returns, not raw prices
    # =========================

    df["Return_Lag_1"] = close.pct_change(1)
    df["Return_Lag_2"] = close.pct_change(2)
    df["Return_Lag_3"] = close.pct_change(3)
    df["Return_Lag_5"] = close.pct_change(5)
    df["Return_Lag_10"] = close.pct_change(10)

    # Volume as relative change instead of raw share counts
    df["Volume_Change_Lag_1"] = df["Volume"].pct_change(1)
    df["Volume_Change_Lag_2"] = df["Volume"].pct_change(2)
    df["Volume_Change_Lag_5"] = df["Volume"].pct_change(5)

    # =========================
    # Returns
    # =========================

    df["Daily_Return"] = close.pct_change()

    df["Log_Return"] = np.log(close / close.shift(1))

    # =========================
    # Volatility
    # =========================

    df["Volatility"] = (
        df["Daily_Return"]
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