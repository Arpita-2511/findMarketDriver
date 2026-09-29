import joblib
import pandas as pd

from sklearn.model_selection import (
    TimeSeriesSplit,
    RandomizedSearchCV
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    Ridge,
    Lasso
)

from sklearn.pipeline import Pipeline

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/final_stock_dataset.csv")

# IMPORTANT (fix applied): drop raw price-level / non-stationary
# columns, matching train_regression.py. Kept features are the
# normalized / return-based engineered features only.
RAW_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

drop_columns = ["Target"]

if "Date" in df.columns:
    drop_columns.append("Date")

for col in RAW_PRICE_COLUMNS:
    if col in df.columns:
        drop_columns.append(col)

X = df.drop(columns=drop_columns)
y = df["Target"]

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(X.columns)}")

# ==========================================
# Time Series Split
# ==========================================

tscv = TimeSeriesSplit(n_splits=3)

# ==========================================
# Ridge Pipeline
# ==========================================

ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

ridge_params = {
    "model__alpha": [
        0.001,
        0.01,
        0.1,
        1,
        10,
        100
    ]
}

# ==========================================
# Lasso Pipeline
# ==========================================

lasso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Lasso(max_iter=50000))
])

lasso_params = {
    "model__alpha": [
        0.001,
        0.01,
        0.1,
        1,
        10
    ]
}

# ==========================================
# Ridge Search
# ==========================================

print("\n")
print("=" * 60)
print("Tuning Ridge Regression")
print("=" * 60)

ridge_search = RandomizedSearchCV(

    estimator=ridge_pipeline,

    param_distributions=ridge_params,

    cv=tscv,

    scoring="r2",

    n_iter=len(ridge_params["model__alpha"]),

    random_state=42,

    n_jobs=-1
)

ridge_search.fit(X, y)

print("\nBest Ridge Parameters")

print(ridge_search.best_params_)

print("\nBest Ridge R²")

print(ridge_search.best_score_)

# ==========================================
# Lasso Search
# ==========================================

print("\n")
print("=" * 60)
print("Tuning Lasso Regression")
print("=" * 60)

lasso_search = RandomizedSearchCV(

    estimator=lasso_pipeline,

    param_distributions=lasso_params,

    cv=tscv,

    scoring="r2",

    n_iter=len(lasso_params["model__alpha"]),

    random_state=42,

    n_jobs=-1
)

lasso_search.fit(X, y)

print("\nBest Lasso Parameters")

print(lasso_search.best_params_)

print("\nBest Lasso R²")

print(lasso_search.best_score_)

# ==========================================
# Save Tuned Models
# ==========================================

joblib.dump(
    ridge_search.best_estimator_,
    "models/ridge_tuned.pkl"
)

joblib.dump(
    lasso_search.best_estimator_,
    "models/lasso_tuned.pkl"
)

print("\n")
print("=" * 60)
print("Saved Models")
print("=" * 60)

print("✔ models/ridge_tuned.pkl")

print("✔ models/lasso_tuned.pkl")