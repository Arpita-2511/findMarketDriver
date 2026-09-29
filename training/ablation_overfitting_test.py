import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ==========================================
# Purpose
# ==========================================
# Isolates which change actually fixed the overfitting gap:
#   A) Original (unregularized) trees, ALL features
#   B) Original (unregularized) trees, REDUCED features only
#   C) Regularized trees, ALL features (no feature reduction)
#   D) Regularized trees, REDUCED features (what you already ran)
#
# For each, we report both TEST R2 (generalization) and the
# TRAIN-minus-TEST R2 gap (a direct measure of overfitting - a large
# positive gap means the model fits training data far better than
# test data, i.e. it's memorizing noise rather than learning
# something that holds up).

df = pd.read_csv("data/final_stock_dataset.csv")

RAW_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
drop_columns = ["Target"]

if "Date" in df.columns:
    drop_columns.append("Date")

for col in RAW_PRICE_COLUMNS:
    if col in df.columns:
        drop_columns.append(col)

X_full = df.drop(columns=drop_columns)
y = df["Target"]

quick_rf = RandomForestRegressor(n_estimators=300, random_state=42)
quick_rf.fit(X_full, y)

TOP_N = 10
top_features = (
    pd.Series(quick_rf.feature_importances_, index=X_full.columns)
    .sort_values(ascending=False)
    .head(TOP_N)
    .index.tolist()
)

X_reduced = X_full[top_features]

N_SPLITS = 20
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

configs = {
    "A) Unregularized, ALL features": {
        "X": X_full,
        "model_params": dict(n_estimators=100, random_state=42),  # original defaults
    },
    "B) Unregularized, REDUCED features": {
        "X": X_reduced,
        "model_params": dict(n_estimators=100, random_state=42),
    },
    "C) Regularized, ALL features": {
        "X": X_full,
        "model_params": dict(
            n_estimators=200, max_depth=3, min_samples_leaf=30,
            max_features=0.5, random_state=42,
        ),
    },
    "D) Regularized, REDUCED features": {
        "X": X_reduced,
        "model_params": dict(
            n_estimators=200, max_depth=3, min_samples_leaf=30,
            max_features=0.5, random_state=42,
        ),
    },
}

print("=" * 80)
print("ABLATION: isolating regularization vs feature reduction")
print("=" * 80)

results = []

for label, cfg in configs.items():

    X_used_full = cfg["X"]
    train_r2_scores = []
    test_r2_scores = []

    for train_idx, test_idx in tscv.split(X_used_full):

        X_train, X_test = X_used_full.iloc[train_idx], X_used_full.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestRegressor(**cfg["model_params"])
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_r2_scores.append(r2_score(y_train, train_preds))
        test_r2_scores.append(r2_score(y_test, test_preds))

    train_r2 = np.mean(train_r2_scores)
    test_r2 = np.mean(test_r2_scores)
    overfit_gap = train_r2 - test_r2

    results.append({
        "config": label,
        "mean_train_R2": train_r2,
        "mean_test_R2": test_r2,
        "overfit_gap (train - test)": overfit_gap,
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\nHow to read this:")
print("- mean_test_R2 tells you actual generalization performance (the only")
print("  number that matters for real use).")
print("- overfit_gap tells you HOW MUCH the model was overfitting. A gap near")
print("  the unregularized configs and a much smaller gap in the regularized")
print("  configs tells you regularization (not feature count) did the real work.")
print("- If B's gap is close to A's gap, feature reduction alone didn't help much.")
print("- If C's gap is close to D's gap, regularization alone was doing the work,")
print("  and you didn't strictly need to drop features.")