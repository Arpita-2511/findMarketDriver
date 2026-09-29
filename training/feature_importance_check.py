import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ==========================================
# Purpose
# ==========================================
# If a tree model's feature importances are all roughly equal (flat),
# it's a sign the model isn't finding any feature meaningfully more
# useful than random noise - consistent with weak/absent signal.
# If a few features clearly stand out, the model IS finding structure,
# even if that structure isn't strong enough to produce a great R2.

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

rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X, y)

xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
xgb.fit(X, y)

rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
xgb_importance = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)

print("=" * 60)
print("RANDOM FOREST - Feature Importances")
print("=" * 60)
print(rf_importance)

print("\n" + "=" * 60)
print("XGBOOST - Feature Importances")
print("=" * 60)
print(xgb_importance)

n_features = len(X.columns)
uniform_importance = 1 / n_features

print("\nNotes:")
print(f"- If every feature were equally (un)informative, importance would be ~{uniform_importance:.4f} each.")
print("- A few features clearly above that baseline = model found some structure.")
print("- All features clustered near that baseline = model isn't finding much beyond noise.")