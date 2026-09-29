import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

df = pd.read_csv("data/final_stock_dataset.csv")

# ==========================================
# Naive baseline for a RETURN target
# ==========================================
# IMPORTANT (fix applied): Target is now next-day return, not raw
# price, so comparing against `Close` (a price level) no longer makes
# sense - it's a different unit entirely. The correct naive baseline
# for a return target is "predict zero return", i.e. assume tomorrow's
# price will be the same as today's (classic random-walk assumption).
# If your real model can't beat this, it isn't adding predictive value.

actual = df["Target"]

prediction = np.zeros_like(actual)

mae = mean_absolute_error(actual, prediction)

rmse = mean_squared_error(
    actual,
    prediction
) ** 0.5

r2 = r2_score(actual, prediction)

print("=" * 60)
print("BASELINE MODEL (naive: predict 0% return / no price change)")
print("=" * 60)

print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")