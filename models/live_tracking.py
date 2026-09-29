import os
import sys
import csv
from datetime import datetime

import joblib

# ==========================================
# Add Project Root to Python Path
# (this file lives in models/, so go up one level to project root,
# same pattern as models/predict.py)
# ==========================================

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.live_stock_service import fetch_latest_stock_data
from features.feature_engineering import engineer_features

# ==========================================
# Purpose
# ==========================================
# Run this once per trading day (e.g. via Task Scheduler / cron, after
# market close). It predicts next-day return with EVERY saved model
# and appends a row per model to live_predictions_log.csv. Nothing is
# scored yet here - you don't know the actual outcome until the next
# trading day. Run evaluate_live_tracking.py periodically (e.g. weekly)
# once actual outcomes are available to see which model is really
# performing well in real time.

SYMBOL = "AAPL"
LOG_FILE = "data/live_predictions_log.csv"
MODELS_DIR = "models/all_models"

RAW_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

feature_order = joblib.load(f"{MODELS_DIR}/feature_order.pkl")

df = fetch_latest_stock_data(SYMBOL)
df = engineer_features(df)

drop_columns = [c for c in ["Date", "Target"] + RAW_PRICE_COLUMNS if c in df.columns]
X = df.drop(columns=drop_columns)
latest_data = X.iloc[[-1]][feature_order]

current_close = float(df.iloc[-1]["Close"])
current_date = str(df.iloc[-1]["Date"])[:10] if "Date" in df.columns else datetime.now().strftime("%Y-%m-%d")

# ==========================================
# Guard against duplicate logging
# ==========================================
# yfinance only updates a symbol's latest daily bar after market close
# (with some lag). If you run this script again before that update
# happens, the "latest" date it sees is still the same as last time -
# without this check, you'd log duplicate rows for the same day.

import pandas as pd

if os.path.isfile(LOG_FILE):
    existing_log = pd.read_csv(LOG_FILE, dtype={"log_date": str})
    already_logged = (
        (existing_log["log_date"] == current_date)
        & (existing_log["symbol"] == SYMBOL)
    ).any()

    if already_logged:
        print(
            f"⚠ {SYMBOL} predictions for {current_date} are already logged. "
            "Yahoo Finance hasn't updated today's bar yet (usually available "
            "a bit after market close) - try again later. Skipping to avoid duplicates."
        )
        sys.exit(0)

model_files = [
    f for f in os.listdir(MODELS_DIR)
    if f.endswith("_model.pkl")
]

rows = []

for model_file in model_files:

    name = model_file.replace("_model.pkl", "")

    model = joblib.load(f"{MODELS_DIR}/{model_file}")
    scaler = joblib.load(f"{MODELS_DIR}/{name}_scaler.pkl")

    X_used = scaler.transform(latest_data) if scaler is not None else latest_data

    predicted_return = float(model.predict(X_used)[0])
    predicted_close = current_close * (1 + predicted_return)

    rows.append({
        "log_date": current_date,
        "symbol": SYMBOL,
        "model": name,
        "current_close": current_close,
        "predicted_return": predicted_return,
        "predicted_close": predicted_close,
        "actual_next_close": "",   # filled in later by evaluate_live_tracking.py
        "actual_return": "",       # filled in later by evaluate_live_tracking.py
    })

    print(f"{name:20s} -> predicted return {predicted_return:+.4%}, predicted close {predicted_close:.2f}")

# ==========================================
# Append to log (create with header if new)
# ==========================================

file_exists = os.path.isfile(LOG_FILE)
os.makedirs("data", exist_ok=True)

with open(LOG_FILE, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    if not file_exists:
        writer.writeheader()
    writer.writerows(rows)

print(f"\n✅ Logged {len(rows)} model predictions for {current_date} to {LOG_FILE}")