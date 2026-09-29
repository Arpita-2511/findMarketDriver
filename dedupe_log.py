import pandas as pd

LOG_FILE = "data/live_predictions_log.csv"

log = pd.read_csv(LOG_FILE, dtype={"log_date": str})

before = len(log)

# Keep only the first occurrence of each (log_date, symbol, model) combo
log = log.drop_duplicates(subset=["log_date", "symbol", "model"], keep="first")

after = len(log)

log.to_csv(LOG_FILE, index=False)

print(f"Removed {before - after} duplicate row(s). {after} rows remain.")
print(log)