import os
import sys

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# Add Project Root to Python Path
# (this file lives in models/, so go up one level to project root,
# same pattern as models/predict.py)
# ==========================================

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==========================================
# Purpose
# ==========================================
# live_tracking.py logs a prediction each day but can't know the
# outcome until the next trading day has actually happened. Run this
# script periodically (e.g. weekly) to:
#   1. Fill in actual outcomes for any log rows where enough time has
#      passed
#   2. Compute real, live (not backtested) performance per model:
#      MAE, RMSE, R2, and directional accuracy (did it call up/down
#      correctly - often the more useful metric for trading)
#
# Only rows with a filled-in actual_return count toward scoring.
#
# Adjustment consistency (Phase 2 fix):
# Prices are dividend/split-adjusted AS OF RETRIEVAL, and Yahoo
# back-adjusts history after every new dividend. The logged
# current_close comes from an older retrieval than today's price
# history, so dividing a freshly fetched close by the logged close
# mixes two price bases. The actual return is therefore computed from
# ONE freshly fetched series: close(next day) / close(log day) - 1.
# actual_next_close is then expressed in the logged price basis
# (current_close * (1 + actual_return)) so it is comparable with
# predicted_close.

LOG_FILE = "data/live_predictions_log.csv"


def compute_outcome(closes: pd.Series, log_date: str, logged_close: float):
    """
    Actual next-day outcome for a prediction logged on `log_date`.

    closes : Close prices indexed by 'YYYY-MM-DD' strings, all from ONE
             retrieval (completed bars only).
    Returns (actual_next_close_in_logged_basis, actual_return), or None
    if the log date or the next trading day is not in `closes`.
    """
    if log_date not in closes.index:
        return None

    dates_after = sorted(d for d in closes.index if d > log_date)
    if not dates_after:
        return None  # next trading day hasn't happened / isn't in the fetched window yet

    next_date = dates_after[0]
    actual_return = float(closes.loc[next_date]) / float(closes.loc[log_date]) - 1
    actual_next_close = float(logged_close) * (1 + actual_return)
    return actual_next_close, actual_return


def fill_outcomes(log: pd.DataFrame, price_history: dict) -> int:
    """Fill actual_next_close / actual_return in place. Returns rows updated."""
    updated = 0

    for idx, row in log.iterrows():

        if pd.notna(row["actual_return"]) and row["actual_return"] != "":
            continue  # already scored

        closes = price_history.get(row["symbol"])
        if closes is None:
            continue

        outcome = compute_outcome(closes, row["log_date"], row["current_close"])
        if outcome is None:
            continue

        log.at[idx, "actual_next_close"], log.at[idx, "actual_return"] = outcome
        updated += 1

    return updated


def score_models(log: pd.DataFrame) -> pd.DataFrame | None:
    scored = log.dropna(subset=["actual_return"])
    scored = scored[scored["actual_return"] != ""]

    if scored.empty:
        return None

    scored = scored.copy()
    scored["actual_return"] = scored["actual_return"].astype(float)
    scored["predicted_return"] = scored["predicted_return"].astype(float)

    results = []

    for model_name, group in scored.groupby("model"):

        n = len(group)
        mae = mean_absolute_error(group["actual_return"], group["predicted_return"])
        rmse = mean_squared_error(group["actual_return"], group["predicted_return"]) ** 0.5

        # R2 needs at least a couple of points and some variance to be meaningful
        r2 = r2_score(group["actual_return"], group["predicted_return"]) if n > 1 else float("nan")

        # Directional accuracy: did the model correctly call up vs down?
        direction_correct = (
            np.sign(group["predicted_return"]) == np.sign(group["actual_return"])
        ).mean()

        results.append({
            "model": model_name,
            "n_predictions": n,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Direction_Accuracy": direction_correct,
        })

    return pd.DataFrame(results).sort_values("Direction_Accuracy", ascending=False)


def main():
    from services.live_stock_service import fetch_latest_stock_data  # network dependency

    log = pd.read_csv(LOG_FILE, dtype={"log_date": str})

    # ==========================================
    # Fetch fresh price history (completed bars only) to look up outcomes
    # ==========================================

    price_history = {}

    for symbol in log["symbol"].unique():
        hist = fetch_latest_stock_data(symbol, period="6mo")
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.strftime("%Y-%m-%d")
        price_history[symbol] = hist.set_index("Date")["Close"]

    updated = fill_outcomes(log, price_history)

    log.to_csv(LOG_FILE, index=False)
    print(f"Filled in {updated} new outcomes.\n")

    # ==========================================
    # Score each model on rows with known outcomes
    # ==========================================

    results_df = score_models(log)

    if results_df is None:
        print("No scored rows yet - check back after a trading day has passed.")
        return

    print("=" * 70)
    print("LIVE MODEL PERFORMANCE (real outcomes, not backtest)")
    print("=" * 70)

    print(results_df.to_string(index=False))

    print("\nNotes:")
    print("- Direction_Accuracy > 0.5 means the model beats a coin flip on up/down calls.")
    print("- With few n_predictions, none of these numbers are reliable yet.")
    print("- Aim for at least 30-60 trading days logged before drawing conclusions.")


if __name__ == "__main__":
    main()
