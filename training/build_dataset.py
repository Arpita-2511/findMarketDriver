import os
import sys

# ==========================================
# Add Project Root to Python Path
# ==========================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from services.live_stock_service import fetch_latest_stock_data
from features.feature_engineering import engineer_features
from features.finalize_dataset import finalize_dataset

# ==========================================
# Configuration
# ==========================================

SYMBOL = "AAPL"
PERIOD = "10y"

print("=" * 50)
print("Building Training Dataset")
print("=" * 50)

# ==========================================
# Step 1 : Download Historical Data
# ==========================================

print(f"\nDownloading {PERIOD} historical data for {SYMBOL}...")

df = fetch_latest_stock_data(
    symbol=SYMBOL,
    period=PERIOD
)

print(f"Downloaded Rows : {len(df)}")

# ==========================================
# Step 2 : Feature Engineering
# ==========================================

print("\nGenerating Technical Indicators...")

df = engineer_features(df)

print("Feature Engineering Completed")

df.to_csv(
    "data/feature_engineered_stock_data.csv",
    index=False
)

# ==========================================
# Step 3 : Final Dataset
# ==========================================

print("\nCreating Target Column...")

final_df = finalize_dataset(df)

print("Target Created")

# ==========================================
# Step 4 : Save Dataset
# ==========================================

final_df.to_csv(
    "data/final_stock_dataset.csv",
    index=False
)

print("\nDataset Saved Successfully!")

print(f"\nDataset Shape : {final_df.shape}")

print("\nColumns:")

print(final_df.columns.tolist())

print("\nFirst Five Rows")

print(final_df.head())