import os
import sys
import joblib


# ==========================================
# Add Project Root to Python Path
# ==========================================

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.live_stock_service import fetch_latest_stock_data
from features.feature_engineering import engineer_features

# ==========================================
# Step 1 : Validate Required Files
# ==========================================

required_files = [
    "models/best_model.pkl",
    "models/scaler.pkl",
    "models/feature_order.pkl"
]

for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"❌ Missing required file: {file}")

print("✅ All model artifacts found!")

# ==========================================
# Step 2 : Load Model Artifacts
# ==========================================

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_order = joblib.load("models/feature_order.pkl")

print("✅ Model Loaded")
print("✅ Scaler Loaded")
print("✅ Feature Order Loaded")

# ==========================================
# Step 3 : User Input
# ==========================================

symbol = input("Enter Stock Symbol : ").strip().upper()

# ==========================================
# Step 4 : Fetch Live Stock Data
# ==========================================

# Returns completed daily bars only (an in-progress session bar is dropped)
df = fetch_latest_stock_data(symbol)

print("✅ Live Stock Data Downloaded")
print(f"   Latest completed bar : {df['Date'].iloc[-1].date()} "
      f"(source: {df.attrs.get('source_method')}, "
      f"incomplete bars dropped: {df.attrs.get('incomplete_bars_dropped')})")

# ==========================================
# Step 5 : Feature Engineering
# ==========================================

df = engineer_features(df)

print("✅ Technical Indicators Generated")

# ==========================================
# Step 6 : Prepare Features
# ==========================================

# IMPORTANT (fix applied): raw OHLCV columns must be excluded from
# the feature matrix, matching how the model was trained. They are
# kept in `df` only so we can read the current Close price below to
# reconstruct a predicted price from the predicted return.
RAW_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

drop_columns = []

if "Date" in df.columns:
    drop_columns.append("Date")

if "Target" in df.columns:
    drop_columns.append("Target")

for col in RAW_PRICE_COLUMNS:
    if col in df.columns:
        drop_columns.append(col)

X = df.drop(columns=drop_columns)

latest_data = X.iloc[[-1]]

# ==========================================
# Step 7 : Feature Order Validation
# ==========================================

required_features = set(feature_order)
received_features = set(latest_data.columns)

missing = required_features - received_features
extra = received_features - required_features

if missing:
    raise ValueError(
        f"❌ Missing Features : {sorted(missing)}"
    )

if extra:
    print(f"⚠ Extra Features Ignored : {sorted(extra)}")

# Reorder exactly as training
latest_data = latest_data[feature_order]

print("✅ Feature validation passed!")

# ==========================================
# Step 8 : Missing Value Validation
# ==========================================

if latest_data.isnull().sum().sum() > 0:
    raise ValueError(
        "❌ NaN values detected before prediction."
    )

print("✅ No missing values detected!")

# ==========================================
# Step 9 : Scale Features
# ==========================================

if scaler is not None:
    latest_scaled = scaler.transform(latest_data)
else:
    # Tree-based best model: no scaling was used during training
    latest_scaled = latest_data

print("✅ Feature scaling completed!")

# ==========================================
# Step 10 : Predict
# ==========================================

# The model now predicts next-day RETURN, not raw price.
predicted_return = float(model.predict(latest_scaled)[0])

current_price = float(df.iloc[-1]["Close"])

# Reconstruct the predicted price from the predicted return
predicted_price = current_price * (1 + predicted_return)

difference = predicted_price - current_price

# ==========================================
# Step 11 : Prediction Sanity Check
# ==========================================

if predicted_price <= 0:
    raise ValueError(
        "❌ Invalid prediction generated."
    )

if predicted_return > 2.0:
    print("⚠ Warning: Predicted return is unusually HIGH.")

if predicted_return < -0.7:
    print("⚠ Warning: Predicted return is unusually LOW.")

# ==========================================
# Step 12 : Trend Direction
# ==========================================

if difference > 0:
    direction = "📈 Bullish"

elif difference < 0:
    direction = "📉 Bearish"

else:
    direction = "➡ Neutral"

percentage_change = predicted_return * 100

# ==========================================
# Step 13 : Final Results
# ==========================================

print("\n==========================================")
print("          Prediction Summary")
print("==========================================")
print(f"Stock Symbol         : {symbol}")
print(f"Current Close Price  : {current_price:.2f}")
print(f"Predicted Next Close : {predicted_price:.2f}")
print(f"Predicted Return     : {predicted_return:.4%}")
print(f"Price Difference     : {difference:.2f}")
print(f"Expected Change      : {percentage_change:.2f}%")
print(f"Market Direction     : {direction}")
print("==========================================")