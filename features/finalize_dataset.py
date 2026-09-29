import pandas as pd
import numpy as np


def finalize_dataset(df):
    """
    Adds the prediction target.

    IMPORTANT (fix applied):
    Target is now next-day RETURN, not next-day raw price.

    Raw price is unbounded and trends over time, so a tree-based model
    trained on historical price levels has no mechanism to output a
    price it never saw during training (it can only average training
    targets). Returns are approximately stationary - the range of
    daily returns stays broadly similar across time - so the range
    seen in training remains representative of future / live data.
    """

    # Handle Date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    else:
        print("No Date column found. Skipping date sorting.")

    df.reset_index(drop=True, inplace=True)

    # Replace inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("\nMissing Values Before Cleaning\n")
    print(df.isnull().sum())

    df.dropna(inplace=True)

    # Create target: next-day percentage return
    # Target_t = (Close_t+1 - Close_t) / Close_t
    df["Target"] = df["Close"].shift(-1) / df["Close"] - 1

    df.dropna(inplace=True)

    df.reset_index(drop=True, inplace=True)

    print("\nMissing Values After Cleaning\n")
    print(df.isnull().sum())

    print("\nFinal Dataset Shape:", df.shape)

    return df