import os
import joblib
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/final_stock_dataset.csv")

print("\n========== Dataset ==========")
print(df.shape)

print("\n========== Target Statistics ==========")
print(df["Target"].describe())

# ==========================================
# Features and Target
# ==========================================

# IMPORTANT (fix applied): drop raw price-level / non-stationary
# columns from the feature matrix. These are unbounded and trend
# over time, which tree models cannot extrapolate beyond. The
# engineered features already capture this information in
# normalized (ratio / return) form.
RAW_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

drop_columns = ["Target"]

if "Date" in df.columns:
    drop_columns.append("Date")

for col in RAW_PRICE_COLUMNS:
    if col in df.columns:
        drop_columns.append(col)

X = df.drop(columns=drop_columns)
y = df["Target"]

print("\n========== Features Used ==========")
print(X.columns.tolist())

# ==========================================
# Time Series Split
# ==========================================

# Using fewer splits gives each training fold
# more historical data and provides a more
# stable evaluation for financial time series.

tscv = TimeSeriesSplit(n_splits=3)

# Models that need feature scaling (distance / gradient-sensitive)
SCALED_MODELS = {
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
}

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=10000),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    ),
}

best_model = None
best_model_name = ""
best_r2 = float("-inf")
best_uses_scaling = False

results = {}

print("\n========== Regression Model Comparison ==========\n")

for name, model in models.items():

    mae_scores = []
    rmse_scores = []
    r2_scores = []

    print(f"\n{name}")
    print("-" * 60)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # ==========================================
        # Feature Scaling (fresh scaler per fold to
        # avoid leaking test-fold statistics)
        # ==========================================

        if name in SCALED_MODELS:
            scaler = StandardScaler()
            X_train_used = scaler.fit_transform(X_train)
            X_test_used = scaler.transform(X_test)
        else:
            # Tree-based models work directly
            # on original feature values.
            X_train_used = X_train
            X_test_used = X_test

        model.fit(X_train_used, y_train)
        predictions = model.predict(X_test_used)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

        print(
            f"Fold {fold} -> "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"R²={r2:.4f}"
        )

    avg_mae = sum(mae_scores) / len(mae_scores)
    avg_rmse = sum(rmse_scores) / len(rmse_scores)
    avg_r2 = sum(r2_scores) / len(r2_scores)

    results[name] = {"MAE": avg_mae, "RMSE": avg_rmse, "R2": avg_r2}

    print("\nAverage Performance")
    print(f"MAE  : {avg_mae:.4f}")
    print(f"RMSE : {avg_rmse:.4f}")
    print(f"R²   : {avg_r2:.4f}")

    if avg_r2 > best_r2:
        best_r2 = avg_r2
        best_model_name = name
        best_uses_scaling = name in SCALED_MODELS

# ==========================================
# Summary of all models
# ==========================================

print("\n========== Summary (sorted by R²) ==========\n")
for name, metrics in sorted(results.items(), key=lambda kv: kv[1]["R2"], reverse=True):
    print(
        f"{name:20s} | "
        f"MAE={metrics['MAE']:.4f} | "
        f"RMSE={metrics['RMSE']:.4f} | "
        f"R²={metrics['R2']:.4f}"
    )

# ==========================================
# Retrain Best Model on Full Dataset
# ==========================================

print(f"\n========== Retraining Best Model ({best_model_name}) on Full Dataset ==========\n")

# Re-instantiate a fresh copy of the best model so it isn't
# left over-fit from cross-validation folds.
best_model = models[best_model_name]

if best_uses_scaling:
    best_scaler = StandardScaler()
    X_final = best_scaler.fit_transform(X)
else:
    best_scaler = None
    X_final = X

best_model.fit(X_final, y)

# ==========================================
# Save Artifacts
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(best_scaler, "models/scaler.pkl")
joblib.dump(X.columns.tolist(), "models/feature_order.pkl")

print("\n========== Best Model ==========\n")
print(f"Best Model : {best_model_name}")
print(f"Average R² : {best_r2:.4f}")

print("\n========== Saved Files ==========\n")
print("✔ models/best_model.pkl")
print("✔ models/scaler.pkl")
print("✔ models/feature_order.pkl")