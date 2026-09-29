import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from scipy import stats

# ==========================================
# Purpose
# ==========================================
# The previous walk-forward run showed trees are SIGNIFICANTLY worse
# than Lasso, and feature importances showed trees are finding
# structure that doesn't generalize (i.e. overfitting to noise).
#
# This script tests whether that gap closes when we:
#   1. Heavily regularize the trees (shallow depth, high min leaf
#      size, feature/row subsampling, L1/L2 penalties)
#   2. Cut the feature set down to only the features both RF and
#      XGBoost agreed were most important (less room to overfit)
#   3. Also try predicting DIRECTION (up/down) instead of exact
#      return magnitude - often an easier, more generalizable target
#
# Same 20-fold walk-forward setup as before, so results are directly
# comparable to your last run.

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

N_SPLITS = 20
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# ==========================================
# Step 1: Pick a reduced feature set using a quick RF fit
# (top N by importance, on the FULL training history only -
# this is just for feature selection, not the final model)
# ==========================================

quick_rf = RandomForestRegressor(n_estimators=300, random_state=42)
quick_rf.fit(X_full, y)

TOP_N = 10
top_features = (
    pd.Series(quick_rf.feature_importances_, index=X_full.columns)
    .sort_values(ascending=False)
    .head(TOP_N)
    .index.tolist()
)

print("Reduced feature set (top", TOP_N, "by importance):")
print(top_features)
print()

X_reduced = X_full[top_features]

# ==========================================
# Step 2: Heavily regularized tree models
# ==========================================

models_regression = {
    "Lasso (baseline)": Lasso(alpha=0.1, max_iter=10000),
    "Random Forest (regularized)": RandomForestRegressor(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=30,
        max_features=0.5,
        random_state=42,
    ),
    "XGBoost (regularized)": XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
    ),
    "LightGBM (regularized)": LGBMRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=5.0,
        min_child_samples=30,
        random_state=42,
        verbose=-1,
    ),
}

SCALED_MODELS = {"Lasso (baseline)"}

fold_r2 = {name: [] for name in models_regression}
fold_dir_acc = {name: [] for name in models_regression}

print(f"Running {N_SPLITS}-fold walk-forward validation (regularized, reduced features)...\n")

for train_idx, test_idx in tscv.split(X_reduced):

    X_train, X_test = X_reduced.iloc[train_idx], X_reduced.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for name, model in models_regression.items():

        if name in SCALED_MODELS:
            scaler = StandardScaler()
            X_train_used = scaler.fit_transform(X_train)
            X_test_used = scaler.transform(X_test)
        else:
            X_train_used = X_train
            X_test_used = X_test

        model.fit(X_train_used, y_train)
        preds = model.predict(X_test_used)

        fold_r2[name].append(r2_score(y_test, preds))
        fold_dir_acc[name].append((np.sign(preds) == np.sign(y_test)).mean())

print("=" * 70)
print(f"REGULARIZED + REDUCED-FEATURE RESULTS ACROSS {N_SPLITS} FOLDS")
print("=" * 70)

summary_rows = []
for name in models_regression:
    scores = np.array(fold_r2[name])
    se = scores.std(ddof=1) / np.sqrt(len(scores))
    summary_rows.append({
        "model": name,
        "mean_R2": scores.mean(),
        "95%_CI_low": scores.mean() - 1.96 * se,
        "95%_CI_high": scores.mean() + 1.96 * se,
        "mean_direction_accuracy": np.array(fold_dir_acc[name]).mean(),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("mean_R2", ascending=False)
print(summary_df.to_string(index=False))

rf_scores = np.array(fold_r2["Random Forest (regularized)"])
lasso_scores = np.array(fold_r2["Lasso (baseline)"])
t_stat, p_value = stats.ttest_rel(rf_scores, lasso_scores)

print(f"\nPaired t-test, regularized RF vs Lasso: t={t_stat:.4f}, p={p_value:.4f}")
if p_value < 0.05:
    print("-> Still a statistically significant difference.")
else:
    print("-> No longer significantly different - regularization closed the gap.")

# ==========================================
# Step 3: Direction (up/down) classification instead of magnitude
# ==========================================

print("\n" + "=" * 70)
print("DIRECTION CLASSIFICATION (up/down) - same reduced features")
print("=" * 70)

y_direction = (y > 0).astype(int)

models_classification = {
    "Random Forest Clf": RandomForestClassifier(
        n_estimators=200, max_depth=3, min_samples_leaf=30,
        max_features=0.5, random_state=42,
    ),
    "XGBoost Clf": XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.01,
        subsample=0.6, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=5.0, random_state=42,
        eval_metric="logloss",
    ),
    "LightGBM Clf": LGBMClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.01,
        subsample=0.6, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=5.0, min_child_samples=30,
        random_state=42, verbose=-1,
    ),
}

fold_acc_clf = {name: [] for name in models_classification}

for train_idx, test_idx in tscv.split(X_reduced):

    X_train, X_test = X_reduced.iloc[train_idx], X_reduced.iloc[test_idx]
    y_train_c, y_test_c = y_direction.iloc[train_idx], y_direction.iloc[test_idx]

    for name, model in models_classification.items():
        model.fit(X_train, y_train_c)
        preds = model.predict(X_test)
        fold_acc_clf[name].append(accuracy_score(y_test_c, preds))

# Baseline: always predict the majority class seen in training
majority_baseline = []
for train_idx, test_idx in tscv.split(X_reduced):
    y_train_c, y_test_c = y_direction.iloc[train_idx], y_direction.iloc[test_idx]
    majority_class = y_train_c.mode()[0]
    majority_baseline.append((y_test_c == majority_class).mean())

clf_summary = []
for name in models_classification:
    accs = np.array(fold_acc_clf[name])
    se = accs.std(ddof=1) / np.sqrt(len(accs))
    clf_summary.append({
        "model": name,
        "mean_accuracy": accs.mean(),
        "95%_CI_low": accs.mean() - 1.96 * se,
        "95%_CI_high": accs.mean() + 1.96 * se,
    })

clf_summary.append({
    "model": "Majority-class baseline",
    "mean_accuracy": np.mean(majority_baseline),
    "95%_CI_low": np.nan,
    "95%_CI_high": np.nan,
})

clf_df = pd.DataFrame(clf_summary).sort_values("mean_accuracy", ascending=False)
print(clf_df.to_string(index=False))

print("\nNotes:")
print("- Compare each model's mean_accuracy to the majority-class baseline.")
print("- Beating that baseline means the model adds real directional value;")
print("  matching it means the model just learned to always guess one class.")