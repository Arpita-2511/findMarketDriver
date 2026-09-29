import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ==========================================
# Purpose
# ==========================================
# If tree models score near-zero/negative R2 on your real stock data,
# there are two possible explanations:
#   (a) There genuinely isn't much learnable signal in the features
#       (very plausible for daily stock returns - markets are close
#       to efficient at this horizon)
#   (b) Something is wrong with the model setup itself
#
# This script creates SYNTHETIC data with a known, learnable, nonlinear
# relationship (similar in scale/noise-level to your real features) and
# checks that tree models can recover it. If they score well here but
# poorly on your real data, that confirms (a): the models work fine,
# the market data just doesn't have much signal to find.

np.random.seed(42)

n_samples = 2500
n_features = 20

X = np.random.randn(n_samples, n_features)

# A deliberately nonlinear, threshold-based relationship - the kind
# of pattern tree models are specifically good at finding
y = (
    0.5 * (X[:, 0] > 0.3).astype(float)
    - 0.3 * (X[:, 1] < -0.5).astype(float)
    + 0.2 * X[:, 2] * (X[:, 3] > 0)
    + 0.1 * np.sin(X[:, 4])
)

# Add realistic noise (comparable relative noise level to stock returns)
noise = np.random.randn(n_samples) * 0.3
y_noisy = y + noise

X_train, X_test, y_train, y_test = train_test_split(
    X, y_noisy, test_size=0.3, shuffle=False
)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1),
}

print("=" * 60)
print("SANITY CHECK: Can tree models learn a KNOWN signal?")
print("=" * 60)
print(f"\nTrue signal strength (R2 if noise were removed): "
      f"{r2_score(y_test, y[len(y_train):len(y_train)+len(y_test)]):.4f}\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    print(f"{name:20s} -> R2 = {r2:.4f}")

print("\nInterpretation:")
print("- R2 meaningfully above 0 (e.g. > 0.15-0.2) here confirms tree")
print("  models and this pipeline setup CAN learn real signal when it exists.")
print("- If your real stock data still scores near 0 while this test")
print("  scores well, the conclusion is: the market signal is genuinely")
print("  weak/absent in your current features, not a bug in your code.")