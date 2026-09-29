import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ==========================================
# Purpose
# ==========================================
# train_regression.py only keeps the single best model by backtest R².
# But backtest R² differences here are tiny and noisy (see fold-by-fold
# variance), so it's too early to trust one winner. This script trains
# and saves EVERY model on the full dataset so live_tracking.py can log
# real-time predictions from all of them side by side. After a few
# weeks of live data, evaluate_live_tracking.py tells you which model
# actually holds up out of sample - not just in a single backtest run.

df = pd.read_csv("data/final_stock_dataset.csv")

RAW_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
drop_columns = ["Target"]

if "Date" in df.columns:
    drop_columns.append("Date")

for col in RAW_PRICE_COLUMNS:
    if col in df.columns:
        drop_columns.append(col)

X = df.drop(columns=drop_columns)
y = df["Target"]

SCALED_MODELS = {"Linear Regression", "Ridge Regression", "Lasso Regression"}

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=10000),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42
    ),
    "LightGBM": LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
}

os.makedirs("models/all_models", exist_ok=True)

joblib.dump(X.columns.tolist(), "models/all_models/feature_order.pkl")

print("=" * 60)
print("Training and saving every model on the full dataset")
print("=" * 60)

for name, model in models.items():

    safe_name = name.lower().replace(" ", "_")

    if name in SCALED_MODELS:
        scaler = StandardScaler()
        X_used = scaler.fit_transform(X)
    else:
        scaler = None
        X_used = X

    model.fit(X_used, y)

    joblib.dump(model, f"models/all_models/{safe_name}_model.pkl")
    joblib.dump(scaler, f"models/all_models/{safe_name}_scaler.pkl")

    print(f"✔ Saved {name}")

print("\nDone. All models saved under models/all_models/")