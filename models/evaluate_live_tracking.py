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

from services.live_stock_service import fetch_latest_stock_data

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

LOG_FILE = "data/live_predictions_log.csv"

log = pd.read_csv(LOG_FILE, dtype={"log_date": str})

symbols = log["symbol"].unique()

# ==========================================
# Fetch fresh price history to look up actual outcomes
# ==========================================

price_history = {}

for symbol in symbols:
    hist = fetch_latest_stock_data(symbol, period="6mo")
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.strftime("%Y-%m-%d")
    price_history[symbol] = hist.set_index("Date")["Close"]

# ==========================================
# Fill in actual outcomes where possible
# ==========================================

updated = 0

for idx, row in log.iterrows():

    if pd.notna(row["actual_return"]) and row["actual_return"] != "":
        continue  # already scored

    symbol = row["symbol"]
    closes = price_history.get(symbol)

    if closes is None:
        continue

    dates_after = sorted(d for d in closes.index if d > row["log_date"])

    if not dates_after:
        continue  # next trading day hasn't happened / isn't in the fetched window yet

    next_date = dates_after[0]
    actual_next_close = float(closes.loc[next_date])
    current_close = float(row["current_close"])
    actual_return = actual_next_close / current_close - 1

    log.at[idx, "actual_next_close"] = actual_next_close
    log.at[idx, "actual_return"] = actual_return
    updated += 1

log.to_csv(LOG_FILE, index=False)
print(f"Filled in {updated} new outcomes.\n")

# ==========================================
# Score each model on rows with known outcomes
# ==========================================

scored = log.dropna(subset=["actual_return"])
scored = scored[scored["actual_return"] != ""]

if scored.empty:
    print("No scored rows yet - check back after a trading day has passed.")
else:
    scored = scored.copy()
    scored["actual_return"] = scored["actual_return"].astype(float)
    scored["predicted_return"] = scored["predicted_return"].astype(float)

    print("=" * 70)
    print("LIVE MODEL PERFORMANCE (real outcomes, not backtest)")
    print("=" * 70)

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

    results_df = pd.DataFrame(results).sort_values("Direction_Accuracy", ascending=False)

    print(results_df.to_string(index=False))

    print("\nNotes:")
    print("- Direction_Accuracy > 0.5 means the model beats a coin flip on up/down calls.")
    print("- With few n_predictions, none of these numbers are reliable yet.")
    print("- Aim for at least 30-60 trading days logged before drawing conclusions.")