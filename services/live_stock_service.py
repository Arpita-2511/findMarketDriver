import time
import pandas as pd
import yfinance as yf


def fetch_latest_stock_data(symbol, period="1y"):
    """
    Fetch historical stock data from Yahoo Finance.

    Parameters
    ----------
    symbol : str
        Stock ticker

    period : str
        Example:
        1mo
        3mo
        6mo
        1y
        2y
        5y

    Returns
    -------
    pandas.DataFrame
    """

    # ==========================================
    # First Attempt : Ticker.history()
    # ==========================================

    try:

        ticker = yf.Ticker(symbol)

        for attempt in range(3):

            df = ticker.history(
                period=period,
                interval="1d",
                auto_adjust=True
            )

            if not df.empty:

                df.reset_index(inplace=True)

                # Remove column name (Price)
                df.columns.name = None

                return df

            print(f"Retry {attempt + 1}/3...")

            time.sleep(2)

    except Exception as e:

        print("Ticker.history() failed:", e)

    # ==========================================
    # Second Attempt : yf.download()
    # ==========================================

    try:

        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if isinstance(df.columns, pd.MultiIndex):

            df.columns = df.columns.get_level_values(0)

        if not df.empty:

            df.reset_index(inplace=True)

            df.columns.name = None

            return df

    except Exception as e:

        print("yf.download() failed:", e)

    # ==========================================
    # Final Failure
    # ==========================================

    raise Exception(
        f"Unable to fetch stock data for '{symbol}'. "
        "Yahoo Finance returned no data."
    )


if __name__ == "__main__":

    symbol = input("Enter Stock Symbol : ").upper()

    stock_df = fetch_latest_stock_data(symbol)

    print(stock_df.tail())

    print("\nColumns:")
    print(stock_df.columns.tolist())