
from features.feature_engineering import add_technical_indicators
from features.finalize_dataset import finalize_dataset

from services.live_stock_service import fetch_latest_stock_data

df = fetch_latest_stock_data(
    symbol="AAPL",
    period="10y"
)

df = add_technical_indicators(df)

df.to_csv(
    "data/feature_engineered_stock_data.csv",
    index=False
)

final_df = finalize_dataset(df)

final_df.to_csv(
    "data/final_stock_dataset.csv",
    index=False
)

print(final_df.head())