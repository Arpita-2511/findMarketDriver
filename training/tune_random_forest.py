import joblib
import pandas as pd

from sklearn.model_selection import (
    TimeSeriesSplit,
    RandomizedSearchCV
)

from sklearn.ensemble import RandomForestRegressor

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/final_stock_dataset.csv")

# IMPORTANT (fix applied): drop raw price-level / non-stationary
# columns, matching train_regression.py. This is especially important
# here since Random Forest is exactly the model type that cannot
# extrapolate beyond price levels seen in training.
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
# Random Forest
# ==========================================

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

# ==========================================
# Hyperparameter Grid
# ==========================================

param_grid = {

    "n_estimators": [
        100,
        200,
        300,
        500
    ],

    "max_depth": [
        5,
        10,
        15,
        20,
        None
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ],

    "max_features": [
        "sqrt",
        "log2",
        None
    ]
}

# ==========================================
# Random Search
# ==========================================

print("\n")
print("=" * 60)
print("Tuning Random Forest")
print("=" * 60)

search = RandomizedSearchCV(

    estimator=rf,

    param_distributions=param_grid,

    n_iter=25,

    cv=tscv,

    scoring="r2",

    random_state=42,

    n_jobs=-1,

    verbose=2

)

search.fit(X, y)

# ==========================================
# Results
# ==========================================

print("\n")
print("=" * 60)
print("Best Parameters")
print("=" * 60)

print(search.best_params_)

print("\nBest R²")

print(search.best_score_)

# ==========================================
# Save Model
# ==========================================

joblib.dump(

    search.best_estimator_,

    "models/random_forest_tuned.pkl"

)

print("\nSaved")

print("✔ models/random_forest_tuned.pkl")