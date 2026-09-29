import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from scipy import stats

# ==========================================
# Purpose
# ==========================================
# 3 backtest folds isn't enough to trust a ranking - Random Forest's
# own R2 swung from +0.02 to -0.07 across folds, larger than the gap
# between models. This script uses many small walk-forward folds
# instead, giving each model a DISTRIBUTION of scores. We can then
# compare distributions properly instead of eyeballing 3 numbers.

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

# Many folds, each with a smaller test window - gives ~20-30 data
# points per model instead of 3
N_SPLITS = 20
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

models = {
    "Lasso": Lasso(alpha=0.1, max_iter=10000),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1),
}

SCALED_MODELS = {"Lasso"}

fold_scores = {name: [] for name in models}
fold_direction_acc = {name: [] for name in models}

print(f"Running {N_SPLITS}-fold walk-forward validation...\n")

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for name, model in models.items():

        if name in SCALED_MODELS:
            scaler = StandardScaler()
            X_train_used = scaler.fit_transform(X_train)
            X_test_used = scaler.transform(X_test)
        else:
            X_train_used = X_train
            X_test_used = X_test

        model.fit(X_train_used, y_train)
        preds = model.predict(X_test_used)

        r2 = r2_score(y_test, preds)
        direction_acc = (np.sign(preds) == np.sign(y_test)).mean()

        fold_scores[name].append(r2)
        fold_direction_acc[name].append(direction_acc)

# ==========================================
# Summary statistics per model
# ==========================================

print("=" * 70)
print(f"R2 ACROSS {N_SPLITS} FOLDS (mean, std, and 95% CI)")
print("=" * 70)

summary_rows = []

for name in models:
    scores = np.array(fold_scores[name])
    mean_r2 = scores.mean()
    std_r2 = scores.std(ddof=1)
    se = std_r2 / np.sqrt(len(scores))
    ci_low, ci_high = mean_r2 - 1.96 * se, mean_r2 + 1.96 * se

    dir_acc = np.array(fold_direction_acc[name])

    summary_rows.append({
        "model": name,
        "mean_R2": mean_r2,
        "std_R2": std_r2,
        "95%_CI_low": ci_low,
        "95%_CI_high": ci_high,
        "mean_direction_accuracy": dir_acc.mean(),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("mean_R2", ascending=False)
print(summary_df.to_string(index=False))

# ==========================================
# Statistical test: is the best tree model meaningfully
# different from the best non-tree model?
# ==========================================

print("\n" + "=" * 70)
print("PAIRED T-TEST: Random Forest vs Lasso (same folds, so paired)")
print("=" * 70)

rf_scores = np.array(fold_scores["Random Forest"])
lasso_scores = np.array(fold_scores["Lasso"])

t_stat, p_value = stats.ttest_rel(rf_scores, lasso_scores)

print(f"Mean R2 - Random Forest : {rf_scores.mean():.4f}")
print(f"Mean R2 - Lasso         : {lasso_scores.mean():.4f}")
print(f"t-statistic             : {t_stat:.4f}")
print(f"p-value                 : {p_value:.4f}")

if p_value < 0.05:
    print("\n-> Statistically significant difference (p < 0.05).")
else:
    print("\n-> NOT statistically significant. The difference you saw")
    print("   in the 3-fold backtest is likely just noise, not a real")
    print("   performance gap between the models.")

print("\nNotes:")
print("- 95% CI overlapping across models means we can't confidently")
print("  say one model is better than another yet.")
print("- mean_direction_accuracy near 0.50 means the model is roughly")
print("  a coin flip on up/down calls - not usable for trading signal.")